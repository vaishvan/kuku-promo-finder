# Promo Finder — framework

Problem 1 of the Kuku brief, treated as a system that gets better with use, not a one-off script.

## 1. What is being built

A service that reads a microdrama screenplay (timestamped events) and returns a ranked list of
20–60s promo clips: episode, ordered segments (start/end), hook, cliffhanger, one-line reason.
It keeps a log of what editors accept, reject or trim, retrains its ranking weights from that log,
and only ships the new weights if an offline evaluation gate says they are better.

Out of scope (per brief): cutting video, music, overlays.

## 2. Assumptions (the brief is silent)

| # | Assumption | Why |
|---|---|---|
| A1 | A promo is judged on hook (first 3s), cliffhanger (ends unresolved), standalone clarity. | Brief's own definition. |
| A2 | Cuts land on script event boundaries (dialogue/action lines), never mid-line. | Timestamps in the pack are per event; mid-line cuts are not defensible. |
| A3 | A promo may be one contiguous stretch or up to 3 stitched segments. | Brief: "can also join a few moments". |
| A4 | No engagement data exists. The only learning signal is editor feedback (accept / reject / trim). | None provided; real CTR arrives only after posting. |
| A5 | Only one show is available, so generalisation to other shows is untested until a new script is run. | Honest limit; eval is split by episode block, not by show. |
| A6 | Sexual or graphic content is flagged and excluded by default. | Platform ad policy risk (Ep 50). |
| A7 | LLM is a swappable judge behind one interface; a deterministic offline judge keeps the system runnable and testable with no key. | Brief: free tiers, model easy to swap. |

## 3. Architecture

```
 script files ──► [1 Parse] ──► [2 Candidate gen] ──► [3 Features] ──► [4 Rank] ──► [5 Select] ──► promos.json
                    code           code (cheap)        code + LLM        learned       code           │
                                                       judge                          (dedupe,       ▼
                                                                                      diversity)   [editor UI/CSV]
                                                                                                      │ accept/reject/trim
                      models/ (versioned weights) ◄── [7 Retrain] ◄── feedback.jsonl ◄───────────────┘
                              ▲                          │
                              └── [8 Gate] ◄── [6 Evaluate on gold, held-out blocks]
```

| Stage | Owner | Why |
|---|---|---|
| 1 Parse | code | Deterministic; the show format is regular. Keep an adapter boundary for other shows. |
| 2 Candidates | code | Sliding windows over event boundaries, 20–60s, plus stitched pairs. Cheap, exhaustive, never invents a timestamp. |
| 3 Features | code + LLM | Code: timing, whole-word lexicons, visual + spoken hook in the first 3 s, story-content cliffhanger (threat/demand/question in the last beats; music cues ignored), character load, weak end-of-episode prior. A clip must show both a hook and a payoff (geometric mean + minimum) to be ranked. LLM: hook / cliffhanger / standalone / emotion on a 0–10 rubric, only for the top-N candidates after a cheap prefilter; the offline stand-in judge never feeds the model. |
| 4 Rank | learned | Logistic ranker over the features. Weights are the thing that is retrained. |
| 5 Select | code | Overlap removal, per-episode cap, diversity across arcs. |
| 6 Evaluate | code | Gold set + hard validity checks + ranking metrics + failure report. |
| 7 Retrain | code | SGD logistic regression, L2 pulled toward the previous weights so one noisy batch cannot swing the model. |
| 8 Gate | code | Champion/challenger: challenger is promoted only if it beats the champion on held-out gold and passes every validity check. |

### Scale and cost
- Parse + candidate gen + code features are linear in script length and need no network.
- Prefilter keeps the top N (default 60) candidates per show for the LLM judge, so LLM cost per show is roughly N calls, not one per window. 50 episodes ≈ 60 calls at about 1.5K tokens each ≈ 90K tokens.
- LLM judge results are cached by (candidate text hash, prompt version, model), so reruns and retrains cost nothing.

### Human in the loop
Editor sees the ranked list with the reason, hook and cliffhanger lines, and accepts, rejects, or trims
start/end. A trim is stored as a positive label for the trimmed window and a negative for the original.

## 4. Learning loop

- **Cold start:** hand-set prior weights (`models/v0000.json`), derived from the brief's definition.
- **Feedback record:** `{candidate_id, show, episode, segments, features, label ∈ {1,0}, editor, ts}`.
- **Retrain:** pointwise logistic regression, loss = log-loss + λ‖w − w_prev‖². Needs ≥ `min_labels` (default 30) new labels.
- **Promotion:** write `models/vNNNN.json`, evaluate vs champion on held-out gold, flip `models/current.json` only if the gate passes. Every run writes a manifest (model version, judge model, prompt hash, config).
- **What does not learn:** the LLM's weights and the rubric text. The rubric is versioned by hand; learned weights decide how much each rubric score matters.

## 5. Evaluation framework

**Gold set** (`gold/gold.jsonl`): promos labelled `must` / `good` / `bad` with segments. Seeded with bootstrap labels
written by the builder (not Kuku editors). They are a starting point and must be reviewed by an editor.

**Hard gates** (any failure blocks promotion):
1. Total duration 20–60s.
2. Every segment boundary is a real event boundary in the script.
3. Segments are in order, within one show, no overlap between returned promos.
4. No safety-flagged content.

**Metrics**
| Metric | What it answers |
|---|---|
| Precision@k, Recall@k vs gold (match = temporal IoU ≥ 0.5) | Does it find the promos an editor would pick? |
| NDCG@k with must=3, good=1, bad=0 | Is the ranking order right? |
| Boundary IoU on matched promos | Are start/end points right? |
| Hook check: independent rule on first 3s | Does the clip actually open on something? |
| Cliffhanger check: ends in last X% of episode or on an unresolved marker | Does it end unresolved? |
| Standalone proxy: unintroduced-character load | Would a new viewer follow? |
| Diversity: distinct episodes / arcs in top-k | Will the feed burn out? |
| Cost / latency per show | Does it scale? |

**Ablations** reported every run: heuristic only → + LLM judge → + learned weights.
**Splits:** by episode block (e.g. Ep 1–30 train, 31–50 held-out), never by candidate, to avoid leakage from overlapping windows.
**Failure report:** top-k picks that gold marks `bad`, with the features that pushed them up. This is the "where does it pick badly" section of the brief.

## 6. Known limits
- One show only; generalisation unproven until the interview's new script.
- Gold labels are bootstrap labels, not editor labels. All metric numbers are relative until reviewed.
- Text only: the screenplay has camera and action lines but no picture, so visual shock (a slap's staging) is inferred from description.
- No real engagement signal. Editor acceptance is a proxy for what performs.
- Offline judge is a lexicon heuristic; the LLM judge path needs an API key and is not exercised without one.
