"""
DNS server that automatically adds any VMs it sees from Proxmox (when that VM has a guest agent installed)
It also is a recursive resolver and will pass any DNS queries that it can't answer upstream
"""
import argparse
import ipaddress
import json
import logging
import os
import re
import socket
import threading
import time
from typing import Dict, List, Optional, Tuple, Any

import requests
from dnslib import DNSRecord, QTYPE, RR, A, PTR
from dnslib.server import DNSServer, BaseResolver, DNSLogger

try:
    import yaml  # optional but recommended
except Exception:
    yaml = None

# -------------------------
# Helpers & parsing
# -------------------------

def normalize_hostname(label: str) -> str:
    s = label.strip().lower()
    s = re.sub(r"[^a-z0-9-]+", "-", s)
    s = s.strip("-")
    return s or "unnamed"

def ensure_fqdn(name: str, domain: str) -> str:
    if name.endswith("."):
        name = name[:-1]
    if "." not in name:
        return f"{name}.{domain}"
    return name

def is_private_v4(ip: str) -> bool:
    try:
        ipobj = ipaddress.ip_address(ip)
        return (isinstance(ipobj, ipaddress.IPv4Address)
                and (ipobj.is_private or ipobj.is_loopback or ipobj.is_link_local))
    except ValueError:
        return False

def pick_primary_ipv4(candidates: List[str], subnet_pref: List[ipaddress.IPv4Network]) -> Optional[str]:
    v4 = [ip for ip in candidates if is_private_v4(ip)]
    if not v4:
        v4 = [ip for ip in candidates if re.match(r"^\d+\.\d+\.\d+\.\d+$", ip)]
    if not v4:
        return None
    if subnet_pref:
        for net in subnet_pref:
            for ip in v4:
                if ipaddress.ip_address(ip) in net:
                    return ip
    return v4[0]

def parse_upstreams(vals: List[Any]) -> List[Tuple[str, int]]:
    out: List[Tuple[str,int]] = []
    for v in vals or []:
        if isinstance(v, str):
            if ":" in v:
                host, port = v.rsplit(":", 1)
                out.append((host.strip(), int(port)))
            else:
                out.append((v.strip(), 53))
        elif isinstance(v, dict):
            host = v.get("host")
            port = int(v.get("port", 53))
            if host:
                out.append((str(host).strip(), port))
        else:
            continue
    # sensible default if empty
    return out or [("1.1.1.1", 53), ("9.9.9.9", 53)]

def load_config(path: str) -> dict:
    """
    Schema (JSON or YAML):
    ---
    domain: home.arpa
    ttl: 60
    refresh_seconds: 60
    listen_addr: 0.0.0.0
    listen_port: 53
    static_overrides_dynamic: true
    subnet_preference: [ "192.168.0.0/16", "10.0.0.0/8", "172.16.0.0/12" ]
    forward:
      enabled: true
      udp_timeout: 2.0
      tcp_timeout: 3.0
      upstreams: ["1.1.1.1", "9.9.9.9:53"]

    proxmox:
      servers:
        - host: "https://pve1.example:8006"
          token_id: "root@pam!mytoken"
          # token_secret may be provided here or via env var named in token_secret_env
          token_secret: "YOUR_TOKEN_SECRET"
          token_secret_env: "PVE1_TOKEN_SECRET"
          verify_ssl: false
        - host: "https://pve2.example:8006"
          token_id: "root@pam!mytoken2"
          token_secret_env: "PVE2_TOKEN_SECRET"
          verify_ssl: false

    static:
      "192.168.1.10": ["nas"]
      "192.168.1.20": ["printer", "hp-printer.home.arpa"]
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) if (path.endswith((".yml", ".yaml")) and yaml) else json.load(f)

    # defaults
    cfg = {
        "domain": "home.arpa",
        "ttl": 60,
        "refresh_seconds": 60,
        "listen_addr": "0.0.0.0",
        "listen_port": 53,
        "static_overrides_dynamic": True,
        "subnet_preference": ["192.168.0.0/16", "10.0.0.0/8", "172.16.0.0/12"],
        "forward": {"enabled": True, "udp_timeout": 2.0, "tcp_timeout": 3.0, "upstreams": ["1.1.1.1", "9.9.9.9"]},
        "proxmox": {"servers": []},
        "static": {}
    }
    # merge
    def deep_merge(a, b):
        for k, v in b.items():
            if isinstance(v, dict) and isinstance(a.get(k), dict):
                deep_merge(a[k], v)
            else:
                a[k] = v
        return a
    cfg = deep_merge(cfg, data or {})

    # process subnet preferences
    cfg["_subnet_preference"] = []
    for s in cfg.get("subnet_preference", []):
        try:
            cfg["_subnet_preference"].append(ipaddress.ip_network(s))
        except Exception:
            pass

    # process upstreams
    fwd = cfg.get("forward", {})
    cfg["_upstreams"] = parse_upstreams(fwd.get("upstreams", []))
    cfg["_udp_timeout"] = float(fwd.get("udp_timeout", 2.0))
    cfg["_tcp_timeout"] = float(fwd.get("tcp_timeout", 3.0))
    cfg["_forward_enabled"] = bool(fwd.get("enabled", True))

    # resolve proxmox token secrets from env if specified
    for s in cfg.get("proxmox", {}).get("servers", []) or []:
        env_key = s.get("token_secret_env")
        if env_key and os.getenv(env_key):
            s["token_secret"] = os.getenv(env_key)

    return cfg

# -------------------------
# Proxmox client
# -------------------------

class ProxmoxClient:
    def __init__(self, host: str, token_id: str, token_secret: str, verify_ssl: bool = True):
        self.base = host.rstrip("/")
        self.headers = {"Authorization": f"PVEAPIToken={token_id}={token_secret}"}
        self.verify_ssl = verify_ssl
        if not verify_ssl:
            requests.packages.urllib3.disable_warnings(  # type: ignore
                category=requests.packages.urllib3.exceptions.InsecureRequestWarning  # type: ignore
            )

    def _get(self, path: str, timeout=10) -> dict:
        url = f"{self.base}/api2/json{path}"
        r = requests.get(url, headers=self.headers, verify=self.verify_ssl, timeout=timeout)
        r.raise_for_status()
        return r.json()

    # --- cluster-wide (preferred) ---
    def list_vms_cluster(self) -> List[dict]:
        """Return items like {'node','vmid','type','name',...} or [] if not permitted."""
        try:
            data = self._get("/cluster/resources?type=vm")
            return data.get("data", []) or []
        except requests.HTTPError:
            return []
        except Exception:
            return []

    # --- per-node fallback ---
    def list_nodes(self) -> List[str]:
        try:
            data = self._get("/nodes").get("data", []) or []
            return [n.get("node") for n in data if n.get("node")]
        except Exception:
            return []

    def list_qemu_on_node(self, node: str) -> List[dict]:
        try:
            arr = self._get(f"/nodes/{node}/qemu").get("data", []) or []
            # normalize: add node/type keys so callers can treat like cluster/resources
            for vm in arr:
                vm.setdefault("node", node)
                vm.setdefault("type", "qemu")
            return arr
        except Exception:
            return []

    def list_lxc_on_node(self, node: str) -> List[dict]:
        try:
            arr = self._get(f"/nodes/{node}/lxc").get("data", []) or []
            for ct in arr:
                ct.setdefault("node", node)
                ct.setdefault("type", "lxc")
            return arr
        except Exception:
            return []

    def list_vms_any(self) -> List[dict]:
        """Try cluster; if empty, enumerate per-node qemu & lxc."""
        vms = self.list_vms_cluster()
        if vms:
            return vms
        out: List[dict] = []
        for node in self.list_nodes():
            out.extend(self.list_qemu_on_node(node))
            out.extend(self.list_lxc_on_node(node))
        return out

    # unchanged helpers to get IPs:
    def get_qemu_ips(self, node: str, vmid: int) -> List[str]:
        try:
            data = self._get(f"/nodes/{node}/qemu/{vmid}/agent/network-get-interfaces")
            ips = []
            for iface in (data.get("data", {}).get("result", []) or []):
                for addr in iface.get("ip-addresses", []) or []:
                    ip = addr.get("ip-address")
                    if ip and ":" not in ip:
                        ips.append(ip)
            return ips
        except Exception as e:
            logging.debug(f"QEMU agent unavailable for {node}/{vmid}: {e}")
            return []

    def get_lxc_ips(self, node: str, vmid: int) -> List[str]:
        ips: List[str] = []
        try:
            cfg = self._get(f"/nodes/{node}/lxc/{vmid}/config").get("data", {})
            for k, v in cfg.items():
                if k.startswith("net") and isinstance(v, str):
                    parts = dict(p.split("=", 1) for p in v.split(",") if "=" in p)
                    if "ip" in parts:
                        ip = parts["ip"].split("/", 1)[0]
                        if re.match(r"^\d+\.\d+\.\d+\.\d+$", ip):
                            ips.append(ip)
        except Exception:
            pass
        try:
            cur = self._get(f"/nodes/{node}/lxc/{vmid}/status/current").get("data", {})
            if "ip" in cur and re.match(r"^\d+\.\d+\.\d+\.\d+$", str(cur["ip"])):
                ips.append(str(cur["ip"]))
            if "ips" in cur and isinstance(cur["ips"], list):
                for ip in cur["ips"]:
                    if isinstance(ip, str) and re.match(r"^\d+\.\d+\.\d+\.\d+$", ip):
                        ips.append(ip)
        except Exception:
            pass
        return list(dict.fromkeys(ips))

# -------------------------
# Mapping store + refresh
# -------------------------

class HostMaps:
    """Thread-safe A/PTR maps."""
    def __init__(self):
        self._lock = threading.RLock()
        self.name_to_ip: Dict[str, str] = {}
        self.ip_to_name: Dict[str, str] = {}

    def replace_all(self, name_to_ip: Dict[str, str], ip_to_name: Dict[str, str]):
        with self._lock:
            self.name_to_ip = name_to_ip
            self.ip_to_name = ip_to_name

    def get_ip(self, fqdn: str) -> Optional[str]:
        with self._lock:
            return self.name_to_ip.get(fqdn.lower())

    def get_name(self, ip: str) -> Optional[str]:
        with self._lock:
            return self.ip_to_name.get(ip)

def build_static_maps(static_cfg: Dict[str, List[str]], domain: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    name_to_ip: Dict[str, str] = {}
    ip_to_name: Dict[str, str] = {}
    for ip, hostnames in (static_cfg or {}).items():
        try:
            ipaddress.ip_address(ip)
        except Exception:
            logging.warning("Skipping invalid IP in static config: %s", ip)
            continue
        for raw in (hostnames or []):
            hn = ensure_fqdn(normalize_hostname(str(raw)), domain)
            name_to_ip[hn] = ip
            ip_to_name.setdefault(ip, hn)  # first name becomes PTR
    return name_to_ip, ip_to_name

def gather_proxmox_maps(cfg: dict) -> Tuple[Dict[str, str], Dict[str, str]]:
    domain = cfg["domain"]
    subnet_pref = cfg["_subnet_preference"]
    name_to_ip: Dict[str, str] = {}
    ip_to_name: Dict[str, str] = {}

    for srv in cfg.get("proxmox", {}).get("servers", []) or []:
        logging.debug("Enumerate proxmox server %s", srv)
        host = srv.get("host"); token_id = srv.get("token_id"); token_secret = srv.get("token_secret")
        if not host or not token_id or not token_secret:
            logging.warning("Skipping Proxmox entry with missing credentials: %s", srv)
            continue
        client = ProxmoxClient(host=host, token_id=token_id, token_secret=token_secret, verify_ssl=bool(srv.get("verify_ssl", True)))
        try:
            vms = client.list_vms_any()
        except Exception as e:
            logging.warning("Failed to list VMs from %s: %s", host, e)
            continue
        logging.debug("Got vms %s", vms)
        for vm in vms:
            name = normalize_hostname(str(vm.get("name", f"vm{vm.get('vmid')}")))
            node = vm.get("node"); vmid = vm.get("vmid"); vtype = vm.get("type")  # 'qemu' or 'lxc'
            logging.debug("Finding IP for %s/%s", node, vmid)
            if node is None or vmid is None:  # safety
                continue

            ips: List[str] = []
            try:
                if vtype == "qemu":
                    ips = client.get_qemu_ips(node, int(vmid))
                elif vtype == "lxc":
                    ips = client.get_lxc_ips(node, int(vmid))
            except Exception as e:
                logging.debug("IP lookup error for %s/%s: %s", node, vmid, e)
            logging.debug("Got IPs for %s/%s: %s", node, vmid, str(ips))

            if not ips:
                continue

            primary = pick_primary_ipv4(ips, subnet_pref)
            if not primary:
                continue

            fqdn = ensure_fqdn(name, domain)
            if fqdn in name_to_ip and name_to_ip[fqdn] != primary:
                logging.info("Hostname collision: %s -> %s replaced by %s", fqdn, name_to_ip[fqdn], primary)
            name_to_ip[fqdn] = primary
            ip_to_name.setdefault(primary, fqdn)

    return name_to_ip, ip_to_name

def refresher_loop(store: HostMaps, cfg_path: str, refresh_sec: int):
    while True:
        try:
            cfg = load_config(cfg_path)

            s_name_to_ip, s_ip_to_name = build_static_maps(cfg.get("static", {}), cfg["domain"])
            d_name_to_ip, d_ip_to_name = gather_proxmox_maps(cfg)

            # precedence
            if cfg.get("static_overrides_dynamic", True):
                merged_name_to_ip = {**d_name_to_ip, **s_name_to_ip}
                merged_ip_to_name = {**d_ip_to_name, **s_ip_to_name}
            else:
                merged_name_to_ip = {**s_name_to_ip, **d_name_to_ip}
                merged_ip_to_name = {**s_ip_to_name, **d_ip_to_name}

            store.replace_all(merged_name_to_ip, merged_ip_to_name)
            logging.info("Refreshed maps: %d hostnames, %d PTRs", len(merged_name_to_ip), len(merged_ip_to_name))
        except Exception as e:
            logging.exception("Refresher loop error: %s", e)

        time.sleep(refresh_sec)

# -------------------------
# DNS Resolver with forwarding
# -------------------------

class MapResolver(BaseResolver):
    def __init__(self, store: HostMaps, cfg: dict):
        self.store = store
        self.domain = cfg["domain"].lower().rstrip(".")
        self.ttl = int(cfg["ttl"])
        self.forward_enabled = bool(cfg["_forward_enabled"])
        self.upstreams = cfg["_upstreams"]
        self.udp_timeout = float(cfg["_udp_timeout"])
        self.tcp_timeout = float(cfg["_tcp_timeout"])

    def _is_local_name(self, qname: str) -> bool:
        return qname == self.domain or qname.endswith("." + self.domain)

    def resolve(self, request: DNSRecord, handler):
        qname = str(request.q.qname).rstrip(".").lower()
        qtype = QTYPE[request.q.qtype]

        # Try answering locally (A)
        if self._is_local_name(qname) and qtype in ("A", "ANY"):
            ip = self.store.get_ip(qname)
            r = request.reply()
            if ip:
                r.add_answer(RR(qname, QTYPE.A, rdata=A(ip), ttl=self.ttl))
                return r
            # else fall through to forward/NX

        # PTR (reverse). If we know it, answer; else forward.
        if qtype == "PTR":
            ip = self._arpa_to_ipv4(qname)
            if ip:
                name = self.store.get_name(ip)
                if name:
                    r = request.reply()
                    r.add_answer(RR(qname, QTYPE.PTR, rdata=PTR(name + "."), ttl=self.ttl))
                    return r
            # else fall through to forward/SERVFAIL

        # For anything we didn't answer locally (including AAAA/MX/TXT/…)
        if self.forward_enabled and self.upstreams:
            raw = self._forward(request.pack())
            if raw:
                try:
                    return DNSRecord.parse(raw)
                except Exception:
                    pass

        # If we got here and it was an A under local domain and we have no record: NXDOMAIN
        if self._is_local_name(qname) and qtype in ("A", "ANY"):
            r = request.reply()
            r.header.rcode = 3
            return r

        # otherwise SERVFAIL
        r = request.reply()
        r.header.rcode = 2
        return r

    @staticmethod
    def _arpa_to_ipv4(qname: str) -> Optional[str]:
        suffix = ".in-addr.arpa"
        if not qname.endswith(suffix):
            return None
        parts = qname[: -len(suffix)].split(".")
        if len(parts) != 4:
            return None
        try:
            ip = ".".join(reversed(parts))
            ipaddress.ip_address(ip)
            return ip
        except ValueError:
            return None

    # ---- forwarding helpers ----

    def _forward(self, query_bytes: bytes) -> Optional[bytes]:
        # UDP first, then TCP fallback per upstream; then next upstream
        for host, port in self.upstreams:
            # UDP try
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                    s.settimeout(self.udp_timeout)
                    s.sendto(query_bytes, (host, port))
                    resp, _ = s.recvfrom(4096)
                # If truncated, retry TCP
                try:
                    if DNSRecord.parse(resp).header.tc:
                        tcp = self._forward_tcp(query_bytes, host, port)
                        if tcp:
                            return tcp
                except Exception:
                    pass
                return resp
            except Exception:
                pass
            # TCP fallback if UDP failed
            try:
                tcp = self._forward_tcp(query_bytes, host, port)
                if tcp:
                    return tcp
            except Exception:
                pass
        return None

    def _forward_tcp(self, query_bytes: bytes, host: str, port: int) -> Optional[bytes]:
        # DNS over TCP uses a 2-byte length prefix
        l = len(query_bytes).to_bytes(2, "big")
        with socket.create_connection((host, port), timeout=self.tcp_timeout) as s:
            s.sendall(l + query_bytes)
            hdr = self._recvn(s, 2)
            if not hdr:
                return None
            n = int.from_bytes(hdr, "big")
            return self._recvn(s, n)

    @staticmethod
    def _recvn(sock: socket.socket, n: int) -> Optional[bytes]:
        buf = b""
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                return None
            buf += chunk
        return buf

# -------------------------
# CLI / Main
# -------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Tiny DNS for statics + Proxmox (A/PTR) with upstream forwarding")
    p.add_argument("--config", default="/etc/pve-dns.yml", help="Path to YAML/JSON config (default: /etc/pve-dns.yml)")
    p.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, ...)")
    return p.parse_args()

def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(args.config)

    store = HostMaps()
    # Build initial maps once
    s_name_to_ip, s_ip_to_name = build_static_maps(cfg.get("static", {}), cfg["domain"])
    d_name_to_ip, d_ip_to_name = gather_proxmox_maps(cfg)

    if cfg.get("static_overrides_dynamic", True):
        name_to_ip = {**d_name_to_ip, **s_name_to_ip}
        ip_to_name = {**d_ip_to_name, **s_ip_to_name}
    else:
        name_to_ip = {**s_name_to_ip, **d_name_to_ip}
        ip_to_name = {**s_ip_to_name, **d_ip_to_name}

    store.replace_all(name_to_ip, ip_to_name)

    # Start refresher thread (hot reloads config + maps)
    t = threading.Thread(target=refresher_loop, args=(store, args.config, int(cfg["refresh_seconds"])), daemon=True)
    t.start()

    resolver = MapResolver(store, cfg)
    logger = DNSLogger(prefix=False)
    server = DNSServer(resolver, port=int(cfg["listen_port"]), address=str(cfg["listen_addr"]), logger=logger)
    logging.info("Starting DNS on %s:%s (domain: .%s, ttl=%ds, refresh=%ds, forward: %s → %s)",
                 cfg["listen_addr"], cfg["listen_port"], cfg["domain"], cfg["ttl"], cfg["refresh_seconds"],
                 "on" if cfg["_forward_enabled"] else "off", cfg["_upstreams"])
    server.start_thread()

    try:
        while server.isAlive():
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Shutting down.")

if __name__ == "__main__":
    main()


