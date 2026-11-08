"""Stage 1: screenplay files -> Episode objects.

Format (one .txt per episode): header lines, then scenes; each scene holds blocks
  **CAMERA:** / **ACTION:** / **SPEAKER (tags):** / **SOUND FX:**
each followed by `Start: (mm:ss.cc) | End: (mm:ss.cc)` and the text.
Show-specific knowledge lives only here. Another show = another parser with the same output.
"""
import glob
import os
import re
from dataclasses import dataclass, field

_TS = re.compile(r"(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)")
_HDR = re.compile(r"^\*\*(.+?)\*\*:?\s*$")
_FLAGS = {"O.S.", "V.O.", "O.S", "V.O", "OFF-SCREEN", "OFF-SCREEN VOICE"}


def ts(s: str) -> float:
    h, m, sec = _TS.search(s).groups()
    return int(h or 0) * 3600 + int(m) * 60 + float(sec)


@dataclass
class Event:
    ep: int
    t0: float
    t1: float
    kind: str  # dialogue | action | camera | sfx
    speaker: str = ""
    tags: list = field(default_factory=list)
    text: str = ""
    scene: int = 0


@dataclass
class Episode:
    ep: int
    logline: str
    end: float
    scene_starts: list  # start time of each scene
    events: list  # Events sorted by t0


def _kind(name: str) -> str:
    n = name.upper()
    if n.startswith("CAMERA"):
        return "camera"
    if n.startswith("ACTION"):
        return "action"
    if n.startswith("SOUND"):
        return "sfx"
    return "dialogue"


def _speaker(name: str):
    base = name.split(" (")[0].strip().upper()
    groups = re.findall(r"\(([^)]*)\)", name)
    tags = []
    for g in groups:
        if g.strip().upper() in _FLAGS:
            continue
        tags += [t.strip() for t in g.split(",") if t.strip() and t.strip().upper() not in _FLAGS]
    return base, tags


def parse_episode(path: str) -> Episode:
    ep = int(re.search(r"episode-(\d+)", os.path.basename(path)).group(1))
    lines = [l.strip() for l in open(path, encoding="utf-8").read().splitlines()]
    logline, events, scene_starts = "", [], []
    scene_end, scene_idx = 0.0, -1
    cur = None  # event being filled

    def flush():
        nonlocal cur
        if cur and cur.text.strip():
            events.append(cur)
        cur = None

    for l in lines:
        if not l:
            continue
        if l.startswith("EPISODE LOGLINE:"):
            logline = l.split(":", 1)[1].strip()
        elif l.startswith("Scene start:"):
            flush()
            scene_idx += 1
            scene_starts.append(ts(l))
        elif l.startswith("Scene end:"):
            scene_end = max(scene_end, ts(l))
        elif l.startswith("Start:") and cur is not None:
            nums = [ts(x) for x in re.findall(r"\(([^)]*)\)", l)]
            if len(nums) >= 2:
                cur.t0, cur.t1 = nums[0], nums[1]
        elif _HDR.match(l):
            flush()
            name = _HDR.match(l).group(1).rstrip(":").strip()
            kind = _kind(name)
            sp, tags = ("", []) if kind != "dialogue" else _speaker(name)
            cur = Event(ep, -1.0, -1.0, kind, sp, tags, "", max(scene_idx, 0))
        elif cur is not None and not l.startswith(("Scene heading:", "SCENE", "CONTINUITY", "---", "Actions and")):
            cur.text += (" " if cur.text else "") + l
    flush()
    events = [e for e in events if e.t0 >= 0]  # drop blocks without timing
    events.sort(key=lambda e: (e.t0, e.t1))
    end = max([scene_end] + [e.t1 for e in events]) if events else scene_end
    return Episode(ep, logline, end, scene_starts, events)


def parse_dir(folder: str) -> list:
    return [parse_episode(p) for p in sorted(glob.glob(os.path.join(folder, "episode-*.txt")))]
