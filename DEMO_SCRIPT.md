# Loom demo script (3 minutes)

The brief asks for a screen recording of up to 3 minutes, in your own voice, showing the solution running.
This script covers the five things Kuku scores: problem choice and framing, approach and scale, quality checking,
the prototype, and communication. Spoken lines are in quotes; screen actions are in brackets. About 420 words.

## Before you hit record

```bash
cd dev/kuku
rm -f runs/feedback.demo.jsonl                     # start the learning-loop tab clean
python -m promofinder --out runs/live serve        # LLM-judged run (cached, costs no API calls)
```

- Open http://127.0.0.1:8765, browser zoom 110%, close other tabs.
- Have a terminal ready in `dev/kuku` for step 5.
- Do not click "Run with LLM judge" on camera: on a new script it spends free-tier requests (50/day).

---

## 0:00 – 0:20 · Problem and why

[Promos tab, top of the list]

"I picked Problem 1: finding promos in a microdrama script. It's the one problem where the input is the real
job: today an editor watches every episode by hand, every week. A promo has to open on a hook in 3 seconds, end
unresolved, and make sense to a stranger. So that's what I built the tool to check."

## 0:20 – 0:55 · How it works

[Point at a card's timecodes, then click Script on the #1 card]

"Code does everything that has to be exact. It parses the screenplay into timed lines, generates about fifteen
thousand candidate windows of 20 to 60 seconds, and only ever cuts on a real line, never mid-sentence. These are
the exact script lines behind this cut. Cheap code features filter that down to about 25 candidates, and only
those go to an LLM judge, scoring hook, cliffhanger and standalone on a rubric. A clip needs both a hook and a
payoff to rank. Here's the judge's own note on this clip [read the "why" line]. It's a free OpenRouter model:
this whole run was 39 calls, zero failures, zero dollars, and swapping the model is one line in a config file."

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
as a test set that the gate never sees. Two honest results. First, a dumb baseline, the last 40 seconds of every
episode, is hard to beat, because the brief defines a promo as ending on a cliffhanger. Second, most of my
model's top picks are moments I never labelled, so they're unknown, not wrong; the tool writes them to a file for
an editor to label. It also tells me what it missed: here, the cathedral explosion in episode 41."

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
