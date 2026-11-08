"""Builds gold/gold.jsonl. BOOTSTRAP labels written by the builder after reading all 50 scripts,
NOT by Kuku editors. Replace/extend with editor labels via the same schema:
{"id","segments":[[ep,start,end]],"tier":"must|good|bad","note"}.

Spec: (tier, ep, mode, seconds, note). mode=tail: last <=N seconds of the episode (cliffhanger endings);
mode=head: first N seconds. Boundaries snap to real events.
"""
import json
import sys
import zlib

sys.path.insert(0, ".")
from promofinder.parse import parse_dir  # noqa: E402

SPEC = [
    # --- must: strong hook + unresolved end, readable cold ---
    ("must", 1, "head", 55, "altar betrayal: 'You should turn around, Juliana'"),
    ("must", 3, "tail", 58, "stranger stops the guards, kneels with a ring"),
    ("must", 11, "tail", 55, "planted watch found in his coat"),
    ("must", 12, "tail", 40, "sister's thorn-garden ultimatum: crawl or he hangs"),
    ("must", 13, "tail", 55, "bare-handed crawl through blackthorn for the ring"),
    ("must", 15, "tail", 45, "thugs trample the garden, husband charges in"),
    ("must", 23, "tail", 55, "arsenic realised, footsteps at the door"),
    ("must", 26, "tail", 40, "Duchess recognises the family ring"),
    ("must", 32, "tail", 50, "branding iron stops at her eye"),
    ("must", 36, "tail", 55, "'their dead Duke is coming home'"),
    ("must", 41, "tail", 45, "wedding, chained bride, cathedral explodes"),
    ("must", 45, "tail", 45, "freed and crowned, then marriage declared void"),
    # --- good ---
    ("good", 1, "tail", 59, "father's silence, 'plainness and low value'"),
    ("good", 2, "tail", 55, "'what your word is worth' public dignity"),
    ("good", 4, "tail", 45, "accepts the vow, disowned, walks out"),
    ("good", 5, "tail", 45, "mother's ledger thrown in the mud"),
    ("good", 9, "tail", 40, "village rumour of a scarred soldier-Duke"),
    ("good", 18, "head", 55, "fire wakes them, escape through the window"),
    ("good", 21, "tail", 45, "'she is dying' - Duchess poisoned"),
    ("good", 22, "tail", 40, "maid disguise, not ordinary fever"),
    ("good", 27, "tail", 45, "framed for poisoning, dragged away"),
    ("good", 28, "tail", 45, "mother's ledger seized, witch accusation"),
    ("good", 31, "tail", 45, "dungeon taunts, iron near the ledger"),
    ("good", 34, "tail", 40, "trial, judge raises his hand"),
    ("good", 35, "tail", 45, "mother finds the 'dead' son"),
    ("good", 42, "tail", 55, "Duke walks in, guards refuse Rupert"),
    ("good", 43, "tail", 45, "Crown names the traitors, Sterling blocked"),
    ("good", 46, "tail", 40, "ten-day ruling, 'ten days is plenty'"),
    # --- bad: quiet, resolved, context-heavy, or unsafe ---
    ("bad", 6, "tail", 45, "quiet cottage, resolved and cosy"),
    ("bad", 7, "tail", 40, "villains flee, resolved"),
    ("bad", 8, "tail", 45, "herb chat, no conflict"),
    ("bad", 16, "tail", 35, "fight over, gaze, resolved"),
    ("bad", 17, "tail", 45, "tender hearth scene, resolved"),
    ("bad", 19, "tail", 40, "barn blanket, 'we begin again'"),
    ("bad", 20, "tail", 40, "villains over tea, no stakes on screen"),
    ("bad", 25, "tail", 50, "talky exposition walk"),
    ("bad", 37, "tail", 40, "invitations and gossip"),
    ("bad", 38, "tail", 40, "bread, bells, resolved prayer"),
    ("bad", 39, "tail", 40, "preparation, no payoff yet"),
    ("bad", 40, "tail", 45, "guests arriving, gossip"),
    ("bad", 47, "tail", 40, "apron accepted, resolved"),
    ("bad", 48, "tail", 40, "scheming in a bedroom, needs context"),
    ("bad", 49, "tail", 55, "secret alliance, needs context, suggestive"),
    ("bad", 50, "head", 60, "sexual content, ad-policy risk"),
    # --- v2 (fix F1): non-tail regions, so the metric stops rewarding "ends at the episode end".
    # Still the same single labeller; see README. mode=range: (start, end) seconds, snapped to events.
    ("must", 3, "range", (66, 110), "stranger walks in: 'Take your hands off her', drops a guard"),
    ("good", 5, "range", (30, 75), "disowned on the steps, ledger flung into the mud"),
    ("must", 11, "head", 50, "accused of stealing the gold watch, constable arrives"),
    ("good", 13, "head", 45, "chained husband begs her not to crawl into the thorns"),
    ("good", 16, "head", 50, "thug swings, husband catches the club bare-handed"),
    ("must", 24, "head", 48, "noblewoman strikes the 'maid' with a riding crop"),
    ("good", 27, "range", (25, 80), "'what is your husband's name?' - doors burst open"),
    ("good", 28, "head", 55, "framed, locket ripped off, identity exposed"),
    ("good", 33, "range", (80, 140), "lord coerces the maid into false testimony"),
    ("good", 36, "range", (60, 120), "mother reveals: arrested, called a witch, condemned"),
    ("must", 42, "head", 45, "'Who are you?' 'You already know.' Guards refuse"),
    ("good", 46, "head", 55, "priest seals the chapel, marriage called a hedge ceremony"),
    ("bad", 6, "range", (40, 95), "cleaning the cottage, quiet"),
    ("bad", 8, "head", 45, "herb identification chat"),
    ("bad", 19, "range", (45, 90), "spy briefing, needs context"),
    ("bad", 20, "head", 50, "villains discuss sugar and succession over tea"),
    ("bad", 25, "head", 50, "slow walk, 'why are you walking like that?'"),
    ("bad", 35, "range", (60, 120), "war-mission exposition over a map"),
    ("bad", 37, "head", 34, "invitations laid on a tray"),
    ("bad", 40, "head", 34, "guests arrive, gossip on the steps"),
]

POSITION = {"tail": "tail", "head": "head", "range": "mid"}


def resolve(ep, mode, n):
    beats = [e for e in ep.events if e.kind != "camera"]
    starts = sorted({e.t0 for e in beats if e.kind in ("dialogue", "action") and not e.text.strip().strip('"').startswith(("...", "…"))})
    ends = sorted({e.t1 for e in beats} | {ep.end})
    if mode == "tail":
        s = next(t for t in starts if ep.end - t <= n + 0.01)
        return s, ep.end
    if mode == "head":
        s = starts[0]
        return s, max(t for t in ends if t <= s + n)
    a, b = n
    s = next(t for t in starts if t >= a)
    return s, max(t for t in ends if t <= min(b, s + 60))


if __name__ == "__main__":
    eps = {e.ep: e for e in parse_dir("data/scripts")}
    with open("gold/gold.jsonl", "w", encoding="utf-8") as f:
        for tier, ep, mode, n, note in SPEC:
            a, b = resolve(eps[ep], mode, n)
            assert 20 <= b - a <= 60.01, (ep, mode, a, b)
            f.write(json.dumps({"id": f"g{ep}{mode[0]}", "segments": [[ep, a, b]], "tier": tier, "position": POSITION[mode],
                                "labeller": "builder", "note": note,
                                # F2: a frozen third of the gold that the promotion gate never sees
                                "split": "test" if zlib.crc32(f"g{ep}{mode[0]}".encode()) % 3 == 0 else "val"}) + "\n")
    print(len(SPEC), "gold items")
