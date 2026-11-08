"""Stage 3a: deterministic features from the script. All in [0,1]. No show-specific names."""
import os
import re
from collections import Counter

CODE_FEATURES = [
    "dur_fit", "hook_arousal", "hook_dialogue", "clean_entry", "cliff_proximity", "cliff_content",
    "cliff_open_end", "resolution", "conflict_density", "emotion_intensity", "speech_ratio",
    "turn_rate", "standalone", "ep_pos", "n_segments", "cross_ep",
]

# Every pattern is anchored on a word start (\b), so "ring" can no longer match "staring" or "Harrington" (fix B2).
# Genre/setting words (married, wife, husband, duke, never, truth) removed from CONFLICT: they occur in nearly
# every scene of this genre and carry no conflict signal (B3).
AROUSAL_TAGS = re.compile(r"\b(shout|scream|furious|shock|gasp|enrag|hysteric|desperat|roar|booming|thunder|outrag|panic|terrif|begging|sob|livid|vicious|menac|threat|wail|frantic|agoni|maniac|defian|stunn)\w*", re.I)
RESOLVE_TAGS = re.compile(r"\b(relie|reassur|tender|peace|warm|content|comfort|gentle|soft|grateful|sincere|fond|playful|amused)\w*", re.I)
HOOK_ACTION = re.compile(
    r"\b(slap(s|ped|ping)?|strikes?|striking|struck|shov(e|es|ed|ing)|slam(s|med|ming)?|grab(s|bed|bing)?|drag(s|ged|ging)?"
    r"|snatch(es|ed|ing)?|fling(s|ing)?|flung|punch(es|ed|ing)?|collaps(e|es|ed|ing)|kneel(s|ing)?|knelt|explo(de|des|ded|ding|sion|sions)"
    r"|shatter(s|ed|ing)?|burst(s|ing)?|blood(y|ied)?|knife|knives|sword|crash(es|ed|ing)?|tackl(e|es|ed|ing)|lunge[sd]?|lunging"
    r"|whip(s|ped|ping)?|branding|chained|gasp(s|ed|ing)?|hurl(s|ed|ing)?)\b", re.I)
CONFLICT = re.compile(
    r"\b(liars?|betray\w*|kill\w*|die|dies|dying|dead|hang|hangs|hanged|gallows|poison\w*|arrest\w*|execut\w*|how dare"
    r"|traitors?|witch\w*|disown\w*|beggar|stole|steal\w*|lie|lies|lying|vengeance|revenge|prison|dungeon|trial|curse[sd]?"
    r"|destroy\w*|burn|burned|burning|impost[eo]r|secret)\b", re.I)
BACKREF = re.compile(r"\b(as i said|as you said|earlier|last time|you remember|what i told|the other day|yesterday|years ago|that night|you promised)\b", re.I)
# Cliffhanger from story content, not from the music cue written at every episode end (fix B1):
# a threat/demand/arrival/reveal/unanswered question in the last beats of the clip.
OPEN_THREAT = re.compile(
    r"\b(who are you|what have you done|where (is|are|did)|what (wedding|do you mean)|or else|unless|until|before (he|she|they|the)"
    r"|you will|you'll|take (her|him) away|seize (her|him)|guards|arrest (her|him)|is coming|are coming|coming home|you're next"
    r"|i'm next|tomorrow|tonight|in two days|in ten days|footsteps|the door|doors? (burst|swing|fly|creak)|bursts? (in|open|through)"
    r"|storms? (in|into)|let's see|do not move|don't move|i know (who|what))\b", re.I)
SAFETY = re.compile(r"\b(shirtless|naked|bare skin|bare chest|carnal|seductive|sexual|undress\w*|forbidden entanglement|passionate embrace)\b", re.I)
# Ad policy (fix H5): Meta bans shocking/excessively violent content, torture and "suffering" imagery in ads.
# Sexual content -> block (never ranked). Graphic violence -> review (ranked, flagged for a human).
VIOLENCE = re.compile(r"\b(blood\w*|branding|searing|whip(s|ped|ping)?|gash\w*|lacerat\w*|tortur\w*|noose|gallows|corpse|flesh"
                      r"|mutilat\w*|knife|knives|sword|bone-rattling|slicing|red-hot|scorch\w*)\b", re.I)
# Spoiler control (fix B7): identity reveals and late-season resolutions.
REVEAL = re.compile(r"\b(it was always me|it's you|you already know|rightful duke|dead duke|declared dead|you're alive|is alive"
                    r"|still alive|my son|true identity|not dead|coming home|you are not .{0,20}wife)\b", re.I)
RELATION = re.compile(r"\b(sister|father|mother|husband|wife|son|daughter|fianc[ée]e?|groom|bride|aunt|cousin|brother|nephew)\b", re.I)
RESOLVE_ACTION = re.compile(r"\b(embrac\w*|hand-in-hand|into the sunlight|peaceful\w*|smil(e|es|ed|ing)|relie(f|ved)|fades? (out|to black)|safe(ly)?|together|rests? (her|his) head|laugh\w*)\b", re.I)
PLATFORM_BANDS = {"any": (25, 55), "tiktok": (20, 35), "reels": (20, 45), "shorts": (25, 55)}  # B10: env PROMO_PLATFORM
HONORIFIC = {"LADY", "LORD", "THE", "SIR", "MR", "MRS", "MISS", "YOUNG", "OLD"}
SPOILER_POS = 0.8  # last 20% of a season is treated as spoiler territory by default


def _schema():
    import hashlib
    src = "|".join(CODE_FEATURES) + "|" + "|".join(p.pattern for p in (AROUSAL_TAGS, RESOLVE_TAGS, HOOK_ACTION, CONFLICT, BACKREF, OPEN_THREAT, SAFETY, VIOLENCE, REVEAL, RELATION, RESOLVE_ACTION)) + "|v2-entry"
    return hashlib.sha256(src.encode()).hexdigest()[:12]


# Stored on every row and feedback label; rows/labels from another schema are refused or skipped (fix A4).
SCHEMA = _schema()
STOP_NAME = {"LADY", "LORD", "THE", "OLD", "YOUNG", "MR", "MRS", "MISS", "SIR", "ELDER", "BLOND", "NOBLE", "ROYAL", "DUKE", "DUCHESS", "DOWAGER", "OFF-SCREEN", "VOICE", "MALE", "FEMALE",
             "MAN", "GOLD", "GOWN", "GROOM", "FORMER", "BEARDED", "MYSTERIOUS", "NOBLEMAN", "CONSTABLE", "GUARD", "PRIEST", "MOTHER", "SCARRED", "HIGH", "POLICE", "CROWN", "OFFICIAL"}


def build_ctx(episodes, aliases=None):
    """Corpus-level facts: ep index, character names, and a speaker-label -> character map (fix B6).
    `aliases` = optional per-show table {"LABEL": "CHAR", "LABEL@ep": "CHAR"}; everything else is automatic."""
    aliases = {k: v for k, v in (aliases or {}).items() if not k.startswith("_")}
    eps = {e.ep: e for e in episodes}
    seen = Counter()
    for e in episodes:
        names = set()
        for ev in e.events:
            if ev.kind == "dialogue":
                for tok in re.findall(r"[A-Z][A-Z-]+", aliases.get(f"{ev.speaker}@{e.ep}", aliases.get(ev.speaker, ev.speaker))):
                    if tok not in STOP_NAME and len(tok) > 2:
                        names.add(tok)
        for n in names:
            seen[n] += 1
    names = {n for n, c in seen.items() if c >= 3}

    def canon(label, ep):
        if f"{label}@{ep}" in aliases:
            return aliases[f"{label}@{ep}"]
        if label in aliases:
            return aliases[label]
        if "'S " in label:  # "ALEXANDER'S MOTHER" is not Alexander
            return label
        toks = [t for t in re.findall(r"[A-Z][A-Z-]+", label) if t not in HONORIFIC]
        return next((t for t in toks if t in names), " ".join(toks) or label)

    from .parse import show_id
    tagged = sum(1 for e in episodes for ev in e.events if ev.kind == "dialogue" and ev.tags)
    dialog = sum(1 for e in episodes for ev in e.events if ev.kind == "dialogue")
    return {"eps": eps, "names": names, "n_eps": max(eps) if eps else 1, "canon": canon, "show": show_id(episodes),
            "tag_coverage": round(tagged / dialog, 3) if dialog else 0.0}


def events_in(episode, t0, t1):
    ev = episode.events
    return [e for e in ev if e.t0 >= t0 - 0.01 and e.t0 < t1 - 0.01]


def _seg_feats(ctx, segments):
    """Return per-candidate feature dict + safety flag + text views for the judge."""
    eps = ctx["eps"]
    seg_events = [(s, events_in(eps[s[0]], s[1], s[2])) for s in segments]
    flat = [e for _, evs in seg_events for e in evs]
    beats = [e for e in flat if e.kind != "camera"]
    dur = sum(s[2] - s[1] for s in segments)
    first_seg, last_seg = segments[0], segments[-1]
    ep0, epl = eps[first_seg[0]], eps[last_seg[0]]
    all_text = " ".join(e.text for e in flat)

    f = {}
    lo, hi = PLATFORM_BANDS.get(os.environ.get("PROMO_PLATFORM", "any"), PLATFORM_BANDS["any"])
    f["dur_fit"] = 1.0 if lo <= dur <= hi else max(0.0, 1 - (min(abs(dur - lo), abs(dur - hi)) / 15))
    # B11: scripts without parenthetical emotion tags fall back to reading the dialogue text itself
    use_text = ctx.get("tag_coverage", 1.0) < 0.5
    hot = lambda e: any(AROUSAL_TAGS.search(t) for t in e.tags) or (use_text and bool(AROUSAL_TAGS.search(e.text)))
    calm = lambda e: any(RESOLVE_TAGS.search(t) for t in e.tags) or (use_text and bool(RESOLVE_TAGS.search(e.text)))

    # hook: first 3s of the clip. Camera text counts for the visual part: the opening frame must work muted.
    head = [e for e in beats if e.t0 < first_seg[1] + 3.0 and e.ep == first_seg[0]]
    head_vis = [e for e in flat if e.t0 < first_seg[1] + 3.0 and e.ep == first_seg[0] and e.kind in ("camera", "action")]
    htxt = " ".join(e.text for e in head)
    vtxt = " ".join(e.text for e in head_vis)
    arousal = sum(1 for e in head if e.kind == "dialogue" and hot(e))
    dtxt = " ".join(e.text for e in head if e.kind == "dialogue")
    f["hook_arousal"] = min(1.0, 0.3 * arousal + 0.25 * min(2, len(HOOK_ACTION.findall(vtxt + " " + htxt))) + 0.15 * dtxt.count("!")
                            + 0.15 * dtxt.count("?") + 0.2 * len(CONFLICT.findall(dtxt)))
    f["hook_dialogue"] = 1.0 if any(e.kind == "dialogue" for e in head) else 0.0
    near_scene = any(abs(first_seg[1] - s) < 0.6 for s in ep0.scene_starts)
    # B9: a scene start is cleanest; a start after a >=1 s pause in the beats is next; a mid-exchange start is weak
    prev_end = max((e.t1 for e in ep0.events if e.kind != "camera" and e.t1 <= first_seg[1] + 0.01), default=None)
    paused = prev_end is None or first_seg[1] - prev_end >= 1.0
    f["clean_entry"] = 1.0 if near_scene else (0.8 if paused else 0.3)

    # cliffhanger: tail of the clip
    gap = epl.end - last_seg[2]
    f["cliff_proximity"] = max(0.0, 1 - gap / 15) if gap >= 0 else 1.0
    tail_ev = [e for e in flat if e.ep == last_seg[0] and e.t1 > last_seg[2] - 3.0 and e.t0 < last_seg[2]]
    # last two story beats (dialogue/action), ignoring camera moves and music cues
    story_tail = [e for e in beats if e.ep == last_seg[0] and e.kind in ("dialogue", "action")][-2:]
    stxt = " ".join(e.text for e in story_tail)
    threat = 1.0 if OPEN_THREAT.search(stxt) or CONFLICT.search(stxt) else 0.0
    question = 1.0 if any(e.kind == "dialogue" and e.text.strip().strip('"').endswith("?") for e in story_tail) else 0.0
    f["cliff_content"] = min(1.0, 0.6 * threat + 0.4 * question)
    tail_dlg = [e for e in tail_ev if e.kind == "dialogue"]
    if tail_dlg:
        last = tail_dlg[-1].text.strip().strip('"')
        # "..." is only an open ending if the speaker does not simply continue in the next line (fix B4)
        nxt = next((e for e in epl.events if e.kind == "dialogue" and e.t0 >= last_seg[2] - 0.01), None)
        continued = last.endswith(("...", "…")) and nxt is not None and nxt.text.strip().strip('"').startswith(("...", "…"))
        trailing = last.endswith(("...", "…", "—")) and not continued
        open_end = 1.0 if (trailing or CONFLICT.search(last)) else (0.7 if last.endswith("?") else 0.0)
    else:
        open_end = 0.3 if tail_ev else 0.0
    f["cliff_open_end"] = open_end
    last3 = [e for e in beats if e.kind == "dialogue"][-3:]
    tone = (sum(1 for e in last3 if calm(e)) / len(last3)) if last3 else 0.0
    # B8: a resolved ending is often only in the action line ("walk hand-in-hand into the sunlight")
    act = " ".join(e.text for e in story_tail if e.kind == "action")
    f["resolution"] = min(1.0, tone + (0.5 if RESOLVE_ACTION.search(act) else 0.0))

    dlg = [e for e in beats if e.kind == "dialogue"]
    f["conflict_density"] = min(1.0, len(CONFLICT.findall(" ".join(e.text for e in dlg))) / max(dur / 10, 1) / 3)
    f["emotion_intensity"] = (sum(1 for e in dlg if hot(e)) / len(dlg)) if dlg else 0.0
    f["speech_ratio"] = min(1.0, sum(e.t1 - e.t0 for e in dlg) / max(dur, 1))
    turns = sum(1 for a, b in zip(dlg, dlg[1:]) if a.speaker != b.speaker)
    f["turn_rate"] = min(1.0, turns / max(dur / 10, 1) / 4)

    canon = ctx.get("canon", lambda s, ep: s)
    speakers = {canon(e.speaker, e.ep) for e in dlg}
    present = {t for s in speakers for t in re.findall(r"[A-Z][A-Z-]+", s)}
    mentioned = {n for n in ctx["names"] if re.search(rf"\b{n}\b", all_text, re.I)}
    unseen = len(mentioned - present)
    load = 0.1 * max(0, len(speakers) - 2) + 0.35 * len(BACKREF.findall(all_text)) + 0.1 * unseen
    # relationship words ("sister", "father", "husband") tell a cold viewer who is who: they help, not hurt
    help_ = 0.15 if RELATION.search(" ".join(e.text for e in dlg)) else 0.0
    f["standalone"] = max(0.0, min(1.0, 1 - load + help_))
    f["ep_pos"] = first_seg[0] / ctx["n_eps"]
    f["n_segments"] = (len(segments) - 1) / 2
    f["cross_ep"] = 1.0 if len({s[0] for s in segments}) > 1 else 0.0

    # views for the judge / explanations
    hook_line = next((e.text for e in head if e.kind == "dialogue"), next((e.text for e in head if e.kind == "action"), htxt[:140]))
    tail_beats = [e for e in beats if e.ep == last_seg[0] and e.t0 >= last_seg[1]]
    cliff_line = next((e.text for e in reversed(tail_beats) if e.kind in ("dialogue", "action")), "")
    safety = bool(SAFETY.search(all_text) or any(SAFETY.search(t) for e in dlg for t in e.tags))
    vio = sorted({m.group(0).lower() for m in VIOLENCE.finditer(all_text)})
    policy = {"label": "block" if safety else ("review" if vio else "none"),
              "reasons": (["sexual content"] if safety else []) + ([f"graphic violence: {', '.join(vio[:4])}"] if vio else [])}
    reveal = bool(REVEAL.search(" ".join(e.text for e in dlg)))
    late = f["ep_pos"] >= SPOILER_POS
    spoiler = "high" if reveal and f["ep_pos"] >= 0.6 else ("medium" if late or reveal else "low")
    # hook type (I7: rotate hook types across a feed)
    if "?" in dtxt:
        hook_type = "question"
    elif HOOK_ACTION.search(vtxt):
        hook_type = "visual spectacle"
    elif arousal or CONFLICT.search(dtxt) or "!" in dtxt:
        hook_type = "emotional conflict"
    else:
        hook_type = "setup"
    lines = []
    for s, evs in seg_events:
        for e in evs:  # camera lines included: the judge must see visual hooks too (fix B5)
            who = f"{e.speaker}: " if e.kind == "dialogue" else f"[{e.kind}] "
            lines.append(f"{e.ep}@{e.t0:05.1f} {who}{e.text}")
    seg_speakers = [sorted({canon(e.speaker, e.ep) for e in evs if e.kind == "dialogue"}) for _, evs in seg_events]
    return f, safety, {"hook_line": hook_line, "cliff_line": cliff_line, "lines": lines, "dur": dur,
                       "policy": policy, "spoiler": spoiler, "hook_type": hook_type,
                       "ep_ends": [eps[s[0]].end for s in segments], "speakers": seg_speakers, "schema": SCHEMA}


def extract(ctx, segments):
    return _seg_feats(ctx, segments)
