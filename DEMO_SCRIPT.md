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
- The page starts empty every time the server starts; results appear only after you upload and run.
- **Once before recording**: upload the zip and click "Run with LLM judge" (NVIDIA free tier, about 5 minutes for
  ~200 clips). Every judgement is cached, so on camera the same upload replays in about 25 seconds. If the judge
  can't run (bad key, model down), the page says so in plain words and offers "Run offline instead".

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
200, including every episode's ending, and only those clips go to an LLM judge, which scores hook, cliffhanger
and standalone. It's NVIDIA's free Nemotron, cached, so a re-run costs nothing; swapping the model is one line in a
config file."

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
every episode, still beats my model, because this show ends every episode on a cliffhanger. The tool told me why:
the two must-haves it missed never made the shortlist, so the judge never saw them. Most of my model's picks are mid-episode moments I never
labelled: unknown, not wrong. So I fixed the shortlist to always include each episode's ending, picked one weight on
the training episodes only, and the gap halved: must-have recall went from zero to half, with no bad picks. The one
it still misses ends on a purely visual beat, which is exactly what the LLM judge is for."

## 2:10 – 2:45 · It learns from the editor

[Learning loop tab. Back on Promos: Accept 4–5 good ones, Reject 2–3 weak ones. Return, click Retrain]

"Every Accept or Reject is a label. Retraining fits new weights, then a gate compares the new model with the
current one on held-out episodes, using a bootstrap interval, and promotes only if the gain is clearly above noise
and it adds no known-bad picks. With a dozen clicks and a thin label set, read this as a demo of the mechanics:
the gate is only as good as its labels."

## 2:45 – 3:00 · Scale, cost, and where AI got it wrong

[Back to Promos]

"At scale, the expensive step is bounded: about 270 judge calls per 50-episode show, five minutes, zero dollars, all cached. I built this with
Claude, and it got things wrong that I had to catch: its first cliffhanger feature fired on 37 of 50 episode
endings just because every episode ends on a music sting, so its first 'improvement' was an artifact. A quality
review found 64 problems; the critical and high ones are fixed and documented."

---

## If you have more time (cut first)
- Show `QUALITY_REVIEW.md` for 5 seconds: problems, evidence, fixes.
- Run offline on the terminal to show it works with no API key:
  `python -m promofinder --out runs/r1 run --offline --k 25` (about 30 seconds).
