#!/usr/bin/env python3
"""
Media file organizer with LLM-based naming using OpenRouter or LMStudio. 

Dependencies:
- requests (for LLM API calls)
- mutagen (optional, for MP3 audiobook detection via ID3 tags)
  Install with: pip install mutagen
  
If mutagen is not installed, MP3 files will be treated as music by default.
"""
import argparse
import hashlib
import json
import os
import re
import sqlite3
import time
import fcntl
from datetime import datetime
from pathlib import Path
import typing as t
import shutil
import subprocess
import requests

try:
    from mutagen.easyid3 import EasyID3
    from mutagen.id3 import ID3
    MUTAGEN_AVAILABLE = True
except ImportError:
    MUTAGEN_AVAILABLE = False

# ===== Defaults / Config =====
DEFAULT_INCOMING = "incoming"
DEFAULT_LIBRARY = "library"
LOCKFILE_NAME = ".media_renamer.lock"

# OpenRouter support: if OPENROUTER_API_KEY is set, use OpenRouter instead of LM Studio
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
if OPENROUTER_API_KEY:
    DEFAULT_API_BASE = "https://openrouter.ai/api/v1"
    DEFAULT_MODEL = os.environ.get("OPENROUTER_MODEL", "google/gemma-3-12b-it")
else:
    DEFAULT_API_BASE = os.environ.get("LMSTUDIO_API_BASE", "http://localhost:1234/v1")
    DEFAULT_MODEL = os.environ.get("LMSTUDIO_MODEL", "qwen2.5-7b-instruct")

SUPPORTED_EXTS = {".mkv", ".mp4", ".avi", ".m4v", ".mov", ".mp3", ".m4b", ".epub", ".azw", ".mobi", ".torrent"}
PROMPT_VERSION = "v1.1.0"  # bump if you change prompts/formatting
DB_NAME = ".media_index.sqlite"

# File type categories
AUDIO_EXTS = {".mp3"}
AUDIOBOOK_EXTS = {".m4b"}
BOOK_EXTS = {".epub", ".azw", ".mobi"}
TORRENT_EXTS = {".torrent"}
VIDEO_EXTS = {".mkv", ".mp4", ".avi", ".m4v", ".mov"}

SYSTEM_PROMPT = """You are a meticulous media filename normalizer for Plex-style libraries.
You must analyze a raw filename and infer either:
- a MOVIE with: movie_title, year (if present)
- a TV episode with: show_title, season (int), episode (int), episode_title (if present)

Rules:
- Only rely on clues in the filename itself (no web lookups).
- Prefer TV if filename matches S01E02 / 1x02 / s1e2 patterns.
- Normalize capitalization (Title Case).
- Integers only for season/episode. Year must be 4 digits if present; omit if unknown (don’t guess).
- Do not include release tags (1080p, WEB, x264, group names) in titles.

Output a single JSON object with keys exactly:
{
  "type": "movie"|"tv",
  "movie_title": string|null,
  "year": int|null,
  "show_title": string|null,
  "season": int|null,
  "episode": int|null,
  "episode_title": string|null
}
No extra text.
"""

USER_PROMPT_TEMPLATE = """Filename: {filename}

Examples:

1) "The.Matrix.1999.1080p.BluRay.x264-GROUP.mkv"
   -> {{"type":"movie","movie_title":"The Matrix","year":1999,"show_title":null,"season":null,"episode":null,"episode_title":null}}

2) "Severance.S01E03.In.Perpetuity.2160p.WEB-DL.mkv"
   -> {{"type":"tv","movie_title":null,"year":null,"show_title":"Severance","season":1,"episode":3,"episode_title":"In Perpetuity"}}

3) "Andor.1x07.Announcement.mkv"
   -> {{"type":"tv","movie_title":null,"year":null,"show_title":"Andor","season":1,"episode":7,"episode_title":"Announcement"}}

Now respond for the given filename only as JSON:
"""

# ===== Utilities =====

class LockFile:
    """Context manager for acquiring an exclusive file lock."""
    def __init__(self, lock_path: Path):
        self.lock_path = lock_path
        self.lock_file = None
    
    def __enter__(self):
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_file = open(self.lock_path, 'w')
        try:
            # Try to acquire an exclusive lock (non-blocking)
            fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            # Write PID to lockfile
            self.lock_file.write(f"{os.getpid()}\n")
            self.lock_file.flush()
            return self
        except IOError:
            self.lock_file.close()
            raise RuntimeError(
                f"Another instance of this script is already running.\n"
                f"Lockfile: {self.lock_path}\n"
                f"If you're sure no other instance is running, delete the lockfile."
            )
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.lock_file:
            fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_UN)
            self.lock_file.close()
            # Clean up the lockfile
            try:
                self.lock_path.unlink()
            except:
                pass

def is_audiobook_mp3(file_path: Path) -> bool:
    """Check if an MP3 file is an audiobook by reading ID3 tags."""
    if not MUTAGEN_AVAILABLE:
        return False
    
    try:
        # Try EasyID3 first for common tags
        audio = EasyID3(str(file_path))
        
        # Check genre tag
        genre = audio.get('genre', [''])[0].lower()
        if 'audiobook' in genre or 'audio book' in genre:
            return True
        
        # Check publisher tag (if available)
        publisher = audio.get('organization', [''])[0].lower()
        if 'audiobook' in publisher or 'audio book' in publisher:
            return True
            
        # Try full ID3 for more detailed tags
        try:
            id3 = ID3(str(file_path))
            # Check TPUB (publisher) frame
            if 'TPUB' in id3:
                pub_text = str(id3['TPUB']).lower()
                if 'audiobook' in pub_text or 'audio book' in pub_text:
                    return True
        except:
            pass
            
    except Exception:
        # If we can't read tags, assume it's not an audiobook
        pass
    
    return False

def count_m4b_siblings(file_path: Path) -> int:
    """Count how many M4B files are in the same parent folder."""
    try:
        parent = file_path.parent
        return sum(1 for f in parent.iterdir() if f.is_file() and f.suffix.lower() == '.m4b')
    except Exception:
        return 1

def sanitize_component(name: str) -> str:
    name = name.strip().replace("/", "-")
    name = name.strip(" .")
    name = re.sub(r"\s+", " ", name)
    return name

def is_media_file(p: Path, exts: t.Set[str]) -> bool:
    return p.is_file() and p.suffix.lower() in exts

def ensure_unique_path(dst: Path) -> Path:
    if not dst.exists():
        return dst
    stem, suffix, parent = dst.stem, dst.suffix, dst.parent
    for i in range(2, 10000):
        cand = parent / f"{stem} ({i}){suffix}"
        if not cand.exists():
            return cand
    return dst

# Fast content signature: size + sha256(first 64KB + last 64KB)
READ_CHUNK = 64 * 1024

def content_signature(path: Path) -> str:
    st = path.stat()
    size = st.st_size
    h = hashlib.sha256()
    with path.open("rb") as f:
        head = f.read(READ_CHUNK)
        h.update(head)
        if size > READ_CHUNK:
            if size > READ_CHUNK * 2:
                f.seek(max(0, size - READ_CHUNK))
                tail = f.read(READ_CHUNK)
            else:
                # read rest if small-ish
                tail = f.read()
            h.update(tail)
    # include size to disambiguate small collisions
    h.update(str(size).encode())
    return f"{size}:{h.hexdigest()}"

def dev_inode(path: Path) -> str:
    st = path.stat()
    # Works on POSIX; Windows: dev/inode may be 0; still OK (secondary key)
    return f"{st.st_dev}:{st.st_ino}"

# ===== LM Studio call =====

def parse_llm_json(s: str) -> dict:
    m = re.search(r"\{.*\}", s, flags=re.S)
    if not m:
        raise ValueError("No JSON object found in LLM output.")
    obj = json.loads(m.group(0))
    for k in ("type","movie_title","year","show_title","season","episode","episode_title"):
        obj.setdefault(k, None)
    ttype = (obj["type"] or "").lower()
    if ttype not in {"movie","tv"}:
        raise ValueError(f"Invalid type: {obj['type']}")
    return obj

def call_lmstudio(api_base: str, model: str, filename: str, temperature: float = 0.2, timeout: int = 60) -> dict:
    url = f"{api_base.rstrip('/')}/chat/completions"
    headers = {"Content-Type": "application/json"}
    
    # Add OpenRouter-specific headers if using OpenRouter
    if OPENROUTER_API_KEY:
        headers["Authorization"] = f"Bearer {OPENROUTER_API_KEY}"
        headers["HTTP-Referer"] = "https://github.com/yourusername/media-renamer"  # Optional but recommended
    
    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role":"system", "content": SYSTEM_PROMPT},
            {"role":"user", "content": USER_PROMPT_TEMPLATE.format(filename=filename)},
        ],
    }
    r = requests.post(url, json=payload, headers=headers, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    content = data["choices"][0]["message"]["content"]
    return parse_llm_json(content)

# ===== Dest path builders =====

def movie_path(library_root: Path, title: str, year: t.Optional[int], ext: str) -> Path:
    title = sanitize_component(title)
    fname = f"{title} ({year}){ext}" if year else f"{title}{ext}"
    return library_root / "movies" / fname

def tv_path(library_root: Path, show: str, season: int, episode: int, ep_title: t.Optional[str], ext: str) -> Path:
    show = sanitize_component(show)
    s, e = f"{season:02d}", f"{episode:02d}"
    fname = f"{show} - s{s}e{e}{(' - ' + sanitize_component(ep_title)) if ep_title else ''}{ext}"
    return library_root / "tv" / show / f"Season {s}" / fname

# ===== Placement (hardlink / copy / reflink) =====

def place_file(src: Path, dst: Path, mode: str) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return f"skip (exists): {dst}"
    if mode == "hard":
        os.link(src, dst)
        return f"hardlinked: {dst} -> {src}"
    elif mode == "copy":
        shutil.copy2(src, dst)
        return f"copied: {dst} <- {src}"
    elif mode == "reflink-auto":
        try:
            subprocess.run(
                ["cp", "--reflink=auto", "--preserve=timestamps", str(src), str(dst)],
                check=True,
            )
            return f"reflink/copied: {dst} <- {src}"
        except Exception:
            shutil.copy2(src, dst)
            return f"copied (fallback): {dst} <- {src}"
    else:
        raise ValueError(f"Unknown link mode: {mode}")

# ===== SQLite index =====

SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
  content_sig   TEXT PRIMARY KEY,
  dev_inode     TEXT,
  size_bytes    INTEGER,
  mtime_ns      INTEGER,
  orig_path     TEXT,
  ext           TEXT,
  meta_json     TEXT,         -- LLM result
  prompt_ver    TEXT,
  final_path    TEXT,         -- where we placed it under library/
  link_mode     TEXT,
  created_at    TEXT,
  updated_at    TEXT
);
CREATE INDEX IF NOT EXISTS idx_files_dev_inode ON files(dev_inode);
CREATE INDEX IF NOT EXISTS idx_files_final_path ON files(final_path);
"""

def open_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    for stmt in SCHEMA.strip().split(";"):
        s = stmt.strip()
        if s:
            conn.execute(s)
    return conn

# ===== Core pipeline =====

def decide_destination(meta: dict, src: Path, library_root: Path, incoming_root: Path) -> Path:
    ext = src.suffix.lower()
    
    # MP3 files: Check if audiobook or music
    if ext in AUDIO_EXTS:
        is_audiobook = is_audiobook_mp3(src)
        
        try:
            rel_path = src.relative_to(incoming_root)
            parent_folder = rel_path.parent
            
            if is_audiobook:
                # MP3 audiobook: preserve folder structure in audiobooks/
                return library_root / "audiobooks" / parent_folder / src.name
            else:
                # Regular music: preserve folder structure in music/
                return library_root / "music" / parent_folder / src.name
        except (ValueError, AttributeError):
            # Fallback if we can't get relative path
            if is_audiobook:
                return library_root / "audiobooks" / src.name
            else:
                return library_root / "music" / src.name
    
    # M4B files: Check if multiple in same folder
    if ext in AUDIOBOOK_EXTS:
        m4b_count = count_m4b_siblings(src)
        
        if m4b_count > 1:
            # Multiple M4B files: preserve folder structure
            try:
                rel_path = src.relative_to(incoming_root)
                parent_folder = rel_path.parent
                return library_root / "audiobooks" / parent_folder / src.name
            except (ValueError, AttributeError):
                # Fallback to flat structure
                return library_root / "audiobooks" / src.name
        else:
            # Single M4B file: flat structure
            return library_root / "audiobooks" / src.name
    
    if ext in BOOK_EXTS:
        return library_root / "books" / src.name
    
    if ext in TORRENT_EXTS:
        return library_root / "torrent-files" / src.name
    
    # Video files use LLM metadata
    if meta["type"] == "movie":
        title = meta.get("movie_title") or Path(src.name).stem
        year = meta.get("year")
        return movie_path(library_root, title, year, ext)
    else:
        show = meta.get("show_title")
        season = meta.get("season")
        episode = meta.get("episode")
        if not (show and isinstance(season, int) and isinstance(episode, int)):
            # fallback unsorted TV
            stem = sanitize_component(Path(src.name).stem) + ext
            return library_root / "tv" / "_Unsorted" / stem
        return tv_path(library_root, show, season, episode, meta.get("episode_title"), ext)

def index_or_place(
    conn: sqlite3.Connection,
    src: Path,
    library_root: Path,
    incoming_root: Path,
    api_base: str,
    model: str,
    link_mode: str,
    dry_run: bool,
    force_llm: bool,
    resignature: bool,
    verbose: bool,
) -> str:
    # Get basic file info (cheap)
    di = dev_inode(src)
    st = src.stat()
    now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    ext = src.suffix.lower()

    # Fast path: Check if we already know this file by dev/inode + mtime (no hashing needed)
    if not resignature and not force_llm:
        cur = conn.execute(
            "SELECT content_sig, meta_json, final_path, link_mode FROM files WHERE dev_inode = ? AND size_bytes = ? AND mtime_ns = ?",
            (di, st.st_size, st.st_mtime_ns)
        )
        row = cur.fetchone()
        if row:
            # File hasn't changed - reuse existing data
            sig, meta_json, final_path, stored_link_mode = row
            meta = json.loads(meta_json) if meta_json else {}
            target = Path(final_path)
            
            if dry_run:
                return f"[DRY-RUN] known (fast): {src.name} -> {target}"
            
            # Only create a new link if the original destination is missing
            if not target.exists():
                action = place_file(src.resolve(), target, mode=link_mode)
            else:
                action = f"skip (exists): {target}"

            # Update orig_path in case file was moved, but keep everything else
            conn.execute(
                "UPDATE files SET orig_path=?, link_mode=?, updated_at=? WHERE content_sig=?",
                (str(src), link_mode, now, sig),
            )
            conn.commit()
            if verbose:
                print(action)
            return f"[FAST] {action}"  # Mark as fast path for logging

    # Slow path: Need to compute content signature (file is new or changed)
    sig = content_signature(src)

    # Look for existing by content_sig
    cur = conn.execute("SELECT meta_json, final_path, link_mode FROM files WHERE content_sig = ?", (sig,))
    row = cur.fetchone()

    if row and not force_llm:
        meta = json.loads(row[0]) if row[0] else {}
        final_path = Path(row[1])
        # Always reuse the original destination path for this content signature
        target = final_path

        if dry_run:
            return f"[DRY-RUN] known: {src.name} -> {target}"
        # Only create a new link if the original destination is missing
        if not target.exists():
            action = place_file(src.resolve(), target, mode=link_mode)
        else:
            action = f"skip (exists): {target}"

        conn.execute(
            "UPDATE files SET dev_inode=?, size_bytes=?, mtime_ns=?, orig_path=?, final_path=?, link_mode=?, updated_at=? WHERE content_sig=?",
            (di, st.st_size, st.st_mtime_ns, str(src), str(target), link_mode, now, sig),
        )
        conn.commit()
        if verbose:
            print(action)
        return action

    # Not known (or force_llm)
    if dry_run:
        needs_llm = ext in VIDEO_EXTS
        return f"[DRY-RUN] NEW -> {'LLM' if needs_llm else 'direct'}: {src.name}"

    # For audio/books/torrents, no LLM needed - just organize by type
    if ext in AUDIO_EXTS or ext in AUDIOBOOK_EXTS or ext in BOOK_EXTS or ext in TORRENT_EXTS:
        meta = {}  # No metadata needed for these
        target = decide_destination(meta, src, library_root, incoming_root)
        
        # Check if target already exists and is the same file (same inode)
        if target.exists():
            src_stat = src.stat()
            target_stat = target.stat()
            if src_stat.st_dev == target_stat.st_dev and src_stat.st_ino == target_stat.st_ino:
                # Already linked - just update the database
                action = f"skip (already linked): {target}"
                conn.execute(
                    "INSERT OR REPLACE INTO files (content_sig, dev_inode, size_bytes, mtime_ns, orig_path, ext, meta_json, prompt_ver, final_path, link_mode, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (sig, di, st.st_size, st.st_mtime_ns, str(src), ext, json.dumps(meta, ensure_ascii=False), PROMPT_VERSION, str(target), link_mode, now, now),
                )
                conn.commit()
                if verbose:
                    print(action)
                return action
        
        target = ensure_unique_path(target)
        action = place_file(src.resolve(), target, mode=link_mode)
        
        conn.execute(
            "INSERT OR REPLACE INTO files (content_sig, dev_inode, size_bytes, mtime_ns, orig_path, ext, meta_json, prompt_ver, final_path, link_mode, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (sig, di, st.st_size, st.st_mtime_ns, str(src), ext, json.dumps(meta, ensure_ascii=False), PROMPT_VERSION, str(target), link_mode, now, now),
        )
        conn.commit()
        if verbose:
            print(action)
        return action

    # Video files need LLM
    try:
        # Pass full relative path from incoming/ to LLM for better context
        try:
            rel_path = str(src.relative_to(library_root.parent / DEFAULT_INCOMING))
        except (ValueError, AttributeError):
            rel_path = src.name
        
        if verbose:
            print(f"  Sending to LLM: {rel_path}")
        
        meta = call_lmstudio(api_base, model, rel_path)
        
        if verbose:
            print(f"  LLM response: {json.dumps(meta, indent=2)}")
            
    except Exception as e:
        # As a safety, put into Unsorted if LLM fails
        meta = {"type":"movie","movie_title":Path(src.name).stem,"year":None,"show_title":None,"season":None,"episode":None,"episode_title":None}
        if verbose:
            print(f"LLM ERROR on {src.name}: {e} -> falling back to movie/unspecific title")

    target = decide_destination(meta, src, library_root, incoming_root)
    
    # Check if target already exists and is the same file (same inode)
    if target.exists():
        src_stat = src.stat()
        target_stat = target.stat()
        if src_stat.st_dev == target_stat.st_dev and src_stat.st_ino == target_stat.st_ino:
            # Already linked - just update the database
            action = f"skip (already linked): {target}"
            conn.execute(
                "INSERT OR REPLACE INTO files (content_sig, dev_inode, size_bytes, mtime_ns, orig_path, ext, meta_json, prompt_ver, final_path, link_mode, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (sig, di, st.st_size, st.st_mtime_ns, str(src), ext, json.dumps(meta, ensure_ascii=False), PROMPT_VERSION, str(target), link_mode, now, now),
            )
            conn.commit()
            if verbose:
                print(action)
            return action
    
    target = ensure_unique_path(target)
    action = place_file(src.resolve(), target, mode=link_mode)

    conn.execute(
        "INSERT OR REPLACE INTO files (content_sig, dev_inode, size_bytes, mtime_ns, orig_path, ext, meta_json, prompt_ver, final_path, link_mode, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (sig, di, st.st_size, st.st_mtime_ns, str(src), ext, json.dumps(meta, ensure_ascii=False), PROMPT_VERSION, str(target), link_mode, now, now),
    )
    conn.commit()
    if verbose:
        print(action)
    return action

def heal_links(conn: sqlite3.Connection, library_root: Path, link_mode: str, dry_run: bool, verbose: bool) -> int:
    """Recreate missing placed files without re-calling LLM."""
    cur = conn.execute("SELECT content_sig, orig_path, final_path FROM files")
    fixed = 0
    for sig, orig, final in cur.fetchall():
        src = Path(orig)
        dst = Path(final) if final else None
        if not dst:
            continue
        if dst.exists():
            continue
        if not src.exists():
            if verbose:
                print(f"missing source; cannot heal: {dst} (from {src})")
            continue
        if dry_run:
            print(f"[DRY-RUN] heal: {dst} -> {src}")
            fixed += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        action = place_file(src.resolve(), dst, mode=link_mode)
        if verbose:
            print(action)
        fixed += 1
    return fixed

def format_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    gb = size_bytes / (1024 ** 3)
    if gb >= 1.0:
        return f"{gb:.2f} GB"
    mb = size_bytes / (1024 ** 2)
    return f"{mb:.2f} MB"

def generate_report(conn: sqlite3.Connection, output_file: Path) -> int:
    """Generate a text report of all indexed files."""
    cur = conn.execute("SELECT orig_path, final_path, size_bytes FROM files ORDER BY final_path")
    rows = cur.fetchall()
    
    with output_file.open('w', encoding='utf-8') as f:
        f.write("=" * 120 + "\n")
        f.write("MEDIA FILE PROCESSING REPORT\n")
        f.write("=" * 120 + "\n\n")
        f.write(f"Total files indexed: {len(rows)}\n")
        f.write(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("=" * 120 + "\n\n")
        
        for orig, final, size in rows:
            f.write(f"Original:  {orig}\n")
            f.write(f"Output:    {final}\n")
            f.write(f"Size:      {format_size(size)}\n")
            f.write("-" * 120 + "\n")
    
    return len(rows)

def generate_html_report(conn: sqlite3.Connection, output_file: Path) -> int:
    """Generate an HTML report of all indexed files."""
    cur = conn.execute("""
        SELECT orig_path, final_path, size_bytes, ext, created_at, updated_at 
        FROM files 
        ORDER BY updated_at DESC
    """)
    rows = cur.fetchall()
    
    # Count by type
    cur_stats = conn.execute("""
        SELECT ext, COUNT(*), SUM(size_bytes) 
        FROM files 
        GROUP BY ext
        ORDER BY COUNT(*) DESC
    """)
    stats = cur_stats.fetchall()
    
    total_size = sum(row[2] for row in rows)
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Media Processing Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
            color: #333;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            margin-top: 0;
            border-bottom: 3px solid #3498db;
            padding-bottom: 15px;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .stat-card h3 {{
            margin: 0 0 10px 0;
            font-size: 14px;
            opacity: 0.9;
            text-transform: uppercase;
        }}
        .stat-card .value {{
            font-size: 32px;
            font-weight: bold;
            margin: 0;
        }}
        .type-stats {{
            margin: 30px 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 14px;
        }}
        th {{
            background: #34495e;
            color: white;
            padding: 12px;
            text-align: left;
            position: sticky;
            top: 0;
            z-index: 10;
        }}
        td {{
            padding: 10px 12px;
            border-bottom: 1px solid #ecf0f1;
        }}
        tr:hover {{
            background: #f8f9fa;
        }}
        .file-path {{
            font-family: 'Monaco', 'Courier New', monospace;
            font-size: 12px;
            color: #555;
            word-break: break-all;
        }}
        .size {{
            text-align: right;
            font-weight: 500;
            color: #2980b9;
        }}
        .timestamp {{
            color: #7f8c8d;
            font-size: 12px;
        }}
        .ext-badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: bold;
            text-transform: uppercase;
            background: #ecf0f1;
            color: #2c3e50;
        }}
        .header-info {{
            background: #ecf0f1;
            padding: 15px;
            border-radius: 6px;
            margin: 20px 0;
        }}
        .search-box {{
            width: 100%;
            padding: 12px;
            margin: 20px 0;
            border: 2px solid #ddd;
            border-radius: 6px;
            font-size: 14px;
            box-sizing: border-box;
        }}
        .search-box:focus {{
            outline: none;
            border-color: #3498db;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Media Processing Report</h1>
        
        <div class="header-info">
            <strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
            <strong>Total Files:</strong> {len(rows):,}<br>
            <strong>Total Size:</strong> {format_size(total_size)}
        </div>
        
        <div class="stats">
            <div class="stat-card">
                <h3>Total Files</h3>
                <p class="value">{len(rows):,}</p>
            </div>
            <div class="stat-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
                <h3>Total Size</h3>
                <p class="value">{format_size(total_size)}</p>
            </div>
            <div class="stat-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
                <h3>File Types</h3>
                <p class="value">{len(stats)}</p>
            </div>
        </div>
        
        <div class="type-stats">
            <h2>📁 Files by Type</h2>
            <table>
                <thead>
                    <tr>
                        <th>Extension</th>
                        <th>Count</th>
                        <th style="text-align: right;">Total Size</th>
                    </tr>
                </thead>
                <tbody>
"""
    
    for ext, count, size in stats:
        html += f"""
                    <tr>
                        <td><span class="ext-badge">{ext}</span></td>
                        <td>{count:,}</td>
                        <td class="size">{format_size(size)}</td>
                    </tr>
"""
    
    html += """
                </tbody>
            </table>
        </div>
        
        <h2>📄 All Files (Most Recent First)</h2>
        <input type="text" class="search-box" id="searchBox" placeholder="🔍 Search files..." onkeyup="filterTable()">
        
        <table id="fileTable">
            <thead>
                <tr>
                    <th>Type</th>
                    <th>Original Path</th>
                    <th>Output Path</th>
                    <th style="text-align: right;">Size</th>
                    <th>Last Updated</th>
                </tr>
            </thead>
            <tbody>
"""
    
    for orig, final, size, ext, created, updated in rows:
        html += f"""
                <tr>
                    <td><span class="ext-badge">{ext}</span></td>
                    <td class="file-path">{orig}</td>
                    <td class="file-path">{final}</td>
                    <td class="size">{format_size(size)}</td>
                    <td class="timestamp">{updated}</td>
                </tr>
"""
    
    html += """
            </tbody>
        </table>
    </div>
    
    <script>
        function filterTable() {
            const input = document.getElementById('searchBox');
            const filter = input.value.toLowerCase();
            const table = document.getElementById('fileTable');
            const rows = table.getElementsByTagName('tr');
            
            for (let i = 1; i < rows.length; i++) {
                const row = rows[i];
                const text = row.textContent.toLowerCase();
                row.style.display = text.includes(filter) ? '' : 'none';
            }
        }
    </script>
</body>
</html>
"""
    
    with output_file.open('w', encoding='utf-8') as f:
        f.write(html)
    
    return len(rows)

# ===== Main =====

def main():
    ap = argparse.ArgumentParser(description="Create Jellyfin-friendly hardlinks/copies using a local LM Studio LLM. Indexes content to avoid re-queries.")
    ap.add_argument("--incoming", default=DEFAULT_INCOMING)
    ap.add_argument("--library", default=DEFAULT_LIBRARY)
    ap.add_argument("--api-base", default=DEFAULT_API_BASE)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--extensions", nargs="*", default=sorted(SUPPORTED_EXTS))
    ap.add_argument("--link-mode", choices=["hard", "copy", "reflink-auto"], default="hard",
                    help="How to mirror into library/: hard (same FS), copy (always), reflink-auto (CoW if supported)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verbose", "-v", action="store_true")
    ap.add_argument("--heal-links", action="store_true", help="Recreate missing targets for already-known files (no LLM calls).")
    ap.add_argument("--force-llm", action="store_true", help="Force re-query LLM even if content is known.")
    ap.add_argument("--resignature", action="store_true", help="Recompute content signatures (normally fast and always done).")
    ap.add_argument("--db", default=None, help="Path to SQLite db (default: incoming/.media_index.sqlite)")
    ap.add_argument("--report", default=None, help="Generate a text report of all indexed files to the specified file path.")
    ap.add_argument("--report-html", action="store_true", help="Generate an HTML report in library/renamer-report.html.")
    ap.add_argument("--daemon", action="store_true", help="Run continuously in daemon mode, processing files repeatedly with a delay between runs.")
    ap.add_argument("--daemon-interval", type=int, default=3600, help="Seconds to wait between daemon runs (default: 3600 = 1 hour).")
    args = ap.parse_args()

    # Warn if mutagen is not available
    if not MUTAGEN_AVAILABLE:
        print("WARNING: mutagen library not found. MP3 audiobook detection disabled.")
        print("Install with: pip install mutagen")
        print("All MP3 files will be treated as music.\n")

    incoming = Path(args.incoming).expanduser().resolve()
    library = Path(args.library).expanduser().resolve()
    exts = {e.lower() if e.startswith(".") else "."+e.lower() for e in args.extensions}
    db_path = Path(args.db) if args.db else incoming / DB_NAME
    lock_path = library / LOCKFILE_NAME

    # Acquire lock to prevent multiple instances
    with LockFile(lock_path):
        conn = open_db(db_path)

        # Report-only mode
        if args.report:
            report_path = Path(args.report)
            count = generate_report(conn, report_path)
            print(f"Report generated: {report_path} ({count} files)")
            return

        # HTML Report-only mode
        if args.report_html:
            report_path = library / "renamer-report.html"
            count = generate_html_report(conn, report_path)
            print(f"HTML report generated: {report_path} ({count} files)")
            return

        # Heal-only mode
        if args.heal_links:
            fixed = heal_links(conn, library, args.link_mode, args.dry_run, args.verbose)
            if args.verbose or args.dry_run:
                print(f"Heal complete; fixed {fixed} items.")
            return

        # Daemon mode: run continuously with interval between runs
        if args.daemon:
            print(f"Starting daemon mode. Will run every {args.daemon_interval} seconds after completion.")
            print(f"Press Ctrl+C to stop.")
            run_count = 0
            try:
                while True:
                    run_count += 1
                    print(f"\n{'='*80}")
                    print(f"DAEMON RUN #{run_count} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"{'='*80}")
                    
                    process_files(conn, incoming, library, exts, args)
                    
                    print(f"\n{'='*80}")
                    print(f"Run #{run_count} complete. Waiting {args.daemon_interval} seconds until next run...")
                    print(f"Next run at: {datetime.fromtimestamp(time.time() + args.daemon_interval).strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"{'='*80}\n")
                    
                    time.sleep(args.daemon_interval)
            except KeyboardInterrupt:
                print(f"\n\nDaemon stopped by user after {run_count} runs.")
                return
        else:
            # Single run mode
            process_files(conn, incoming, library, exts, args)

def process_files(conn, incoming, library, exts, args):
    """Process all files in the incoming directory."""
    # Recursively find media files
    files = [p for p in incoming.rglob("*") if is_media_file(p, exts)]
    total_files = len(files)
    print(f"Found {total_files} media files in {incoming}")

    processed = 0
    fast_matches = 0
    
    for src in sorted(files):
        try:
            msg = index_or_place(
                conn=conn,
                src=src,
                library_root=library,
                incoming_root=incoming,
                api_base=args.api_base,
                model=args.model,
                link_mode=args.link_mode,
                dry_run=args.dry_run,
                force_llm=args.force_llm,
                resignature=args.resignature,
                verbose=args.verbose,
            )
            processed += 1
            
            # Track fast matches separately
            is_fast_match = msg.startswith("[FAST]")
            if is_fast_match:
                fast_matches += 1
            
            # Only log non-fast matches (slow hash lookups, new files, LLM calls)
            if not is_fast_match or args.verbose:
                remaining = total_files - processed
                progress_msg = f"[{processed}/{total_files}] Remaining: {remaining} | {src.name}"
                print(progress_msg)
                if args.verbose:
                    print(f"  -> {msg}")
        except Exception as e:
            print(f"ERROR processing {src}: {e}")

    db_path = Path(args.db) if args.db else incoming / DB_NAME
    print(f"\nDone. Processed {processed} files ({fast_matches} fast cache hits). DB: {db_path}")

if __name__ == "__main__":
    main()
