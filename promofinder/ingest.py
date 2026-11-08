"""Safe intake of a scripts zip: extract only .txt files into a fresh folder, named episode-NNN-*.txt.

The zip is untrusted input. Member paths are never used to build output paths (no zip-slip), only .txt files are
read, and size/count limits stop zip bombs. Episode numbers come from the filename, else the script's own
"EPISODE NUMBER:" header, else file order.
"""
import os
import re
import zipfile

MAX_FILES = 1000
MAX_FILE_BYTES = 5 * 1024 * 1024
MAX_TOTAL_BYTES = 100 * 1024 * 1024


def _decode(b):
    for enc in ("utf-8-sig", "cp1252"):
        try:
            return b.decode(enc)
        except UnicodeDecodeError:
            continue
    return b.decode("utf-8", errors="replace")


def _episode_no(name, text):
    m = re.search(r"episode[-_ ]?(\d+)", name, re.I) or re.search(r"EPISODE NUMBER:\s*Episode\s*(\d+)", text, re.I)
    return int(m.group(1)) if m else None


def extract_scripts(zip_path, dest):
    """Returns (written_paths, skipped[(member, reason)])."""
    os.makedirs(dest, exist_ok=True)
    written, skipped, total, used = [], [], 0, set()
    with zipfile.ZipFile(zip_path) as z:
        members = [i for i in z.infolist() if not i.is_dir()]
        if len(members) > MAX_FILES:
            raise ValueError(f"zip has {len(members)} files; limit is {MAX_FILES}")
        pending = []
        for info in members:
            base = os.path.basename(info.filename.replace("\\", "/"))
            if not base.lower().endswith(".txt"):
                skipped.append((info.filename, "not a .txt file"))
                continue
            if base.startswith("._") or "__MACOSX" in info.filename:
                skipped.append((info.filename, "macOS metadata"))
                continue
            if info.file_size > MAX_FILE_BYTES:
                skipped.append((info.filename, f"larger than {MAX_FILE_BYTES // 1024 // 1024} MB"))
                continue
            total += info.file_size
            if total > MAX_TOTAL_BYTES:
                raise ValueError(f"zip expands past {MAX_TOTAL_BYTES // 1024 // 1024} MB")
            with z.open(info) as f:
                data = f.read(MAX_FILE_BYTES + 1)
            if len(data) > MAX_FILE_BYTES:  # header lied about the size
                skipped.append((info.filename, "larger than declared"))
                continue
            text = _decode(data)
            pending.append((base, text, _episode_no(base, text)))
    nxt = max([n for _, _, n in pending if n] or [0]) + 1
    for base, text, n in sorted(pending, key=lambda t: (t[2] is None, t[2] or 0, t[0])):
        if n is None or n in used:
            n, nxt = nxt, nxt + 1
        used.add(n)
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", os.path.splitext(base)[0])[:60]
        path = os.path.join(dest, f"episode-{n:03d}-{safe}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        written.append(path)
    if not written:
        raise ValueError("no .txt scripts found in the zip")
    return written, skipped
