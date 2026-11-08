# Loom demo script (3 minutes)

The brief asks for a screen recording of up to 3 minutes, in your own voice, showing the solution running.
This script covers the five things Kuku scores: problem choice and framing, approach and scale, quality checking,
the prototype, and communication. Spoken lines are in quotes; screen actions are in brackets. About 420 words.

## Before you hit record

```bash
cd dev/kuku
rm -f runs/feedback.demo.jsonl          # start the learning-loop tab clean
python -m promofinder serve             # http://127.0.0.1:8765
```

- Have the scripts zip ready in a file explorer window next to the browser (drag-and-drop looks best on camera).
- Browser zoom 100–110%, close other tabs. The header shows how many free LLM requests are left today.
- Every clip the judge has already scored is cached, so uploading the same zip again costs nothing and runs in
  about 20 seconds. A different zip spends real requests: about 27 for scoring plus up to 12 for pairwise
  re-ranking (skipped automatically if the day's quota is too low). Record on a day with 15+ requests left if you
  want to show a fresh judgement, or just re-run the cached one.

---

## 0:00 – 0:20 · Problem and why

[Upload screen, lilac block]

"I picked Problem 1: finding promos in a microdrama script. It's the one problem where the input is the real
job: today an editor watches every episode by hand, every week. A promo has to open on a hook in 3 seconds, end
unresolved, and make sense to a stranger. So that's what I built the tool to check."

## 0:20 – 0:55 · Live run: zip in, promos out

[Drag the zip onto the drop zone. "50 scripts extracted" appears. Click "Run with LLM judge". Talk over the
pipeline block as the steps tick]

"I drop in the scripts zip exactly as Kuku sent it. It's extracted on my machine, only the text files. Code does
everything that has to be exact: it parses every scene into timed lines and generates about fifteen thousand
candidate cuts of 20 to 60 seconds, always on a real line, never mid-sentence. Cheap code features shortlist about
25, and only those clips are sent to an LLM judge, which scores hook, cliffhanger and standalone. A free OpenRouter
model, cached, so a re-run costs nothing; swapping the model is one line in a config file."

[Results appear. Click Script on the #1 card]

"These are the exact script lines behind this cut, and here's the judge's own note on it." [read the "why" line]

## 0:55 – 1:30 · The output an editor gets

[Scroll one card: hook type, policy badge, "why". Then the Posting plan tab]

"Every promo comes with its timecodes, plus handles for the editor, the opening line, the ending line, and why it
ranked. Two flags matter for ads: this one is marked for ad-policy review because Meta restricts graphic violence,
and late-season reveals are marked as spoilers. The brief asks what to post first. The posting plan leads with
premise clips for cold audiences, rotates hook types because these creatives fatigue in days, and holds back the
flagged ones for a human."

## 1:30 – 2:10 · How I know it's good, and where it fails

[Evaluation tab]

"I hand-labelled regions across all 50 episodes, held out the last 20 episodes, and froze a third of the labels
as a test set that the gate never sees. Honest result: on those labels a dumb baseline, the last 40 seconds of
every episode, beats my model, because this show ends every episode on a cliffhanger. The tool tells me why: the
two must-haves it missed, like the cathedral explosion in episode 41, never made the 27-clip shortlist I could
afford on the free tier, so the judge never saw them. And most of my model's picks are mid-episode moments I never
labelled: unknown, not wrong. They go to a file for an editor. The fix is clear: always shortlist each episode's
ending, and judge the full pool."

## 2:10 – 2:45 · It learns from the editor

[Learning loop tab. Back on Promos: Accept 4–5 good ones, Reject 2–3 weak ones. Return, click Retrain]

"Every Accept or Reject is a label. Retraining fits new weights, then a gate compares the new model with the
current one on held-out episodes, using a bootstrap interval, and promotes only if the gain is clearly above noise
and it adds no known-bad picks. With a dozen clicks and a thin label set, read this as a demo of the mechanics:
the gate is only as good as its labels."

## 2:45 – 3:00 · Scale, cost, and where AI got it wrong

[Back to Promos]

"At scale, the expensive step is bounded: about 40 judge calls per 50-episode show, all cached. I built this with
Claude, and it got things wrong that I had to catch: its first cliffhanger feature fired on 37 of 50 episode
endings just because every episode ends on a music sting, so its first 'improvement' was an artifact. A quality
review found 64 problems; the critical and high ones are fixed and documented."

---

## If you have more time (cut first)
- Show `QUALITY_REVIEW.md` for 5 seconds: problems, evidence, fixes.
- Run offline on the terminal to show it works with no API key:
  `python -m promofinder --out runs/r1 run --offline --k 25` (about 30 seconds).
