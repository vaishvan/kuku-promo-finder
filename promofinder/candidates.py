"""Stage 2: enumerate candidate promos. Pure code, exhaustive, cuts only on real event boundaries."""
from bisect import bisect_left

TARGETS_PROMO = (20, 25, 30, 35, 40, 45, 50, 55, 60)
TARGETS_SEG = (6, 10, 14, 18, 22)


def cid(segments) -> str:
    return "+".join(f"{ep}:{a:.2f}-{b:.2f}" for ep, a, b in segments)


def windows(episode, lo, hi, targets, include_end=True):
    """Yield (ep, t0, t1): start = a dialogue/action start, end = a beat end, duration in [lo, hi]."""
    beats = [e for e in episode.events if e.kind != "camera"]
    dlg = [e for e in beats if e.kind == "dialogue"]
    cont = lambda e: e.text.strip().strip('"').startswith(("...", "…"))
    # never start on the second half of a split line ("...let her go."), never end right before one
    cut_before = {a.t1 for a, b in zip(dlg, dlg[1:]) if cont(b)}
    starts = sorted({e.t0 for e in beats if e.kind in ("dialogue", "action") and not (e.kind == "dialogue" and cont(e))})
    ends = sorted({e.t1 for e in beats} - cut_before)
    seen = set()
    for s in starts:
        picks = set()
        for t in targets:
            i = bisect_left(ends, s + t)
            for j in (i - 1, i):
                if 0 <= j < len(ends) and lo <= ends[j] - s <= hi:
                    picks.add(ends[j])
        if include_end and lo <= episode.end - s <= hi:
            picks.add(episode.end)
        for e in picks:
            key = (round(s, 2), round(e, 2))
            if key not in seen:
                seen.add(key)
                yield (episode.ep, s, e)


def singles(episodes):
    return [[w] for ep in episodes for w in windows(ep, 20, 60, TARGETS_PROMO)]


def short_segments(episodes):
    return [w for ep in episodes for w in windows(ep, 6, 22, TARGETS_SEG)]


def stitches(heads, tails, max_pairs=3000):
    """Pair a hook-heavy head with a cliffhanger tail. heads/tails are (ep,t0,t1); total 20-60s, no overlap."""
    out = []
    for h in heads:
        for t in tails:
            # same episode only: cross-episode pairs joined unrelated storylines (review C4)
            if h == t or h[0] != t[0] or (h[1] < t[2] and t[1] < h[2]):
                continue
            if 20 <= (h[2] - h[1]) + (t[2] - t[1]) <= 60:
                # order within an episode follows the script; across episodes head first
                pair = sorted([h, t]) if h[0] == t[0] else [h, t]
                out.append(pair)
                if len(out) >= max_pairs:
                    return out
    return out
