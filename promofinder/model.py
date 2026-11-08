"""Stage 4: the learned ranker + versioned registry. Pure python (20 features, a few thousand rows)."""
import json
import math
import os
import random

from .features import CODE_FEATURES

JUDGE_FEATURES = ["j_hook", "j_cliff", "j_standalone", "j_emotion"]
FEATURES = CODE_FEATURES + JUDGE_FEATURES + ["hook_and_payoff"]
MODELS = "models"
# Not learnable from feedback: ep_pos (feedback comes from whichever episodes editors reviewed, so it would
# learn "early episodes are good"), n_segments / cross_ep (single-segment feedback carries no signal on them).
FROZEN = {"ep_pos", "n_segments", "cross_ep"}
MIN_PART = 0.1  # a promo needs BOTH some hook and some payoff; below this on either it is not ranked (fix C1)

# v0 prior: hand-set, NOT fitted (E6). Provenance: hook/cliffhanger/standalone weights come from the brief's
# definition of a promo; hook_and_payoff 2.0 encodes "both required" (I2: high hook + low hold = clickbait);
# resolution -1.0 because a resolved ending is the opposite of a cliffhanger; judge weights 1.5 assume an LLM judge
# is more reliable than lexicons (unvalidated until editor labels exist); ep_pos -0.3 is a mild spoiler prior.
# Retraining moves these from feedback; they are the starting point, not a claim.
# cliff_proximity is now a weak prior only: ending at the episode end is not the same as ending unresolved (B1).
PRIOR = {
    "bias": -4.0,
    "w": {"dur_fit": 0.5, "hook_arousal": 1.0, "hook_dialogue": 0.5, "clean_entry": 0.5, "cliff_proximity": 0.3,
          "cliff_content": 1.0, "cliff_open_end": 0.8, "resolution": -1.0, "conflict_density": 0.8,
          "emotion_intensity": 0.6, "speech_ratio": 0.3, "turn_rate": 0.2, "standalone": 1.0, "ep_pos": -0.3,
          "n_segments": -0.4, "cross_ep": -0.6, "j_hook": 1.5, "j_cliff": 1.5, "j_standalone": 1.0, "j_emotion": 0.8,
          "hook_and_payoff": 2.0},
}


def real_judge(judge) -> bool:
    """The offline judge reuses the feature lexicons, so feeding its scores to the model double counts (fix D1).
    Only a real (LLM) judge contributes j_* features; otherwise they stay neutral."""
    return bool(judge) and not str(judge.get("by", "")).startswith("offline")


def vec(feats: dict, judge: dict | None) -> dict:
    x = dict(feats)
    j = judge if real_judge(judge) else {}
    x["j_hook"] = j.get("hook", 5) / 10
    x["j_cliff"] = j.get("cliffhanger", 5) / 10
    x["j_standalone"] = j.get("standalone", 5) / 10
    x["j_emotion"] = j.get("emotion", 5) / 10
    # hook AND payoff: geometric mean, so a huge hook cannot carry a flat ending (the "clickbait" pattern, C1)
    hook = feats.get("hook_arousal", 0.0)
    pay = max(feats.get("cliff_content", 0.0), feats.get("cliff_open_end", 0.0))
    if j:
        hook, pay = (hook + x["j_hook"]) / 2, (pay + x["j_cliff"]) / 2
    x["_hook_part"], x["_payoff_part"] = round(hook, 4), round(pay, 4)
    x["hook_and_payoff"] = math.sqrt(hook * pay)
    return x


class Model:
    def __init__(self, w=None, bias=None, version="v0000", meta=None):
        self.w = dict(w or PRIOR["w"])
        self.bias = PRIOR["bias"] if bias is None else bias
        self.version, self.meta = version, meta or {}

    def logit(self, x):
        return self.bias + sum(self.w.get(k, 0.0) * x.get(k, 0.0) for k in FEATURES)

    def score(self, x):
        return 1 / (1 + math.exp(-self.logit(x)))

    def contributions(self, x, n=3):
        c = sorted(((self.w.get(k, 0.0) * x.get(k, 0.0), k) for k in FEATURES), reverse=True)
        return [k for v, k in c[:n] if v > 0]

    def to_json(self):
        return {"version": self.version, "bias": self.bias, "w": self.w, "meta": self.meta}


def train(rows, labels, prior: Model, l2=1.0, lr=0.05, epochs=200, seed=0, weights=None):
    """Class-balanced log-loss + l2 * ||w - w_prior||^2 (per-sample, so l2 is the strength vs ONE example).
    A small noisy batch cannot drag the model far from the champion."""
    rnd = random.Random(seed)
    w, b = dict(prior.w), prior.bias
    free = [k for k in FEATURES if k not in FROZEN]
    idx = list(range(len(rows)))
    pos = sum(labels)
    cw = {1: 0.5 * len(labels) / max(pos, 1), 0: 0.5 * len(labels) / max(len(labels) - pos, 1)}
    for _ in range(epochs):
        rnd.shuffle(idx)
        for i in idx:
            x, y = rows[i], labels[i]
            z = b + sum(w.get(k, 0.0) * x.get(k, 0.0) for k in FEATURES)
            g = cw[y] * (weights[i] if weights else 1.0) * (1 / (1 + math.exp(-max(-30, min(30, z)))) - y)
            pull = min(0.5, lr * l2 * 0.01)  # clamped: a huge l2 must mean "stay at prior", not overshoot
            for k in free:
                w[k] -= lr * g * x.get(k, 0.0)
                w[k] -= pull * (w[k] - prior.w.get(k, 0.0))
            b -= lr * g
    return Model(w, b)


def recency_weights(records, half_life_days=90.0, now=None):
    """E5: a label loses half its weight every `half_life_days`; records without a timestamp count fully."""
    import time
    now = now or time.time()
    return [0.5 ** (max(0.0, now - r["ts"]) / 86400 / half_life_days) if r.get("ts") else 1.0 for r in records]


def pairs_of(labels, groups):
    """(accepted, rejected) index pairs that an editor saw in the same group (episode)."""
    by = {}
    for i, g in enumerate(groups):
        by.setdefault(g, []).append(i)
    return [(i, j) for ix in by.values() for i in ix for j in ix if labels[i] == 1 and labels[j] == 0]


def train_pairwise(rows, labels, groups, prior: Model, l2=1.0, lr=0.05, epochs=200, seed=0, weights=None):
    """E3: ranking loss log(1 + exp(-(s_pos - s_neg))) over accepted-vs-rejected pairs from the same episode.
    Editors choose between alternatives, so the label that matters is the order, not an absolute probability.
    The bias cancels in a difference, so it is kept from the prior."""
    P = pairs_of(labels, groups)
    if not P:
        return train(rows, labels, prior, l2=l2, lr=lr, epochs=epochs, seed=seed, weights=weights)
    rnd = random.Random(seed)
    w = dict(prior.w)
    free = [k for k in FEATURES if k not in FROZEN]
    for _ in range(epochs):
        rnd.shuffle(P)
        for i, j in P:
            xi, xj = rows[i], rows[j]
            d = sum(w.get(k, 0.0) * (xi.get(k, 0.0) - xj.get(k, 0.0)) for k in FEATURES)
            g = -1 / (1 + math.exp(max(-30, min(30, d))))  # d/dd of log(1+exp(-d))
            if weights:
                g *= (weights[i] + weights[j]) / 2
            pull = min(0.5, lr * l2 * 0.01)
            for k in free:
                w[k] -= lr * g * (xi.get(k, 0.0) - xj.get(k, 0.0))
                w[k] -= pull * (w[k] - prior.w.get(k, 0.0))
    return Model(w, prior.bias)


def auc(model, rows, labels):
    """Probability a random positive outranks a random negative (label-only, no gold needed)."""
    s = [model.logit(x) for x in rows]
    pos = [a for a, y in zip(s, labels) if y]
    neg = [a for a, y in zip(s, labels) if not y]
    if not pos or not neg:
        return None
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


# ---- registry: models/vNNNN.json + models/current.json ----
# Writes go to a temp file and are swapped in with os.replace (atomic on one filesystem), under a lock file, so a
# crash cannot leave a half-written current.json and two retrains cannot claim the same version (fix A5).
def _write_json(path, obj):
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def _read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class _Lock:
    def __init__(self, timeout=30):
        self.path, self.timeout = os.path.join(MODELS, ".lock"), timeout

    def __enter__(self):
        import time
        t0 = time.time()
        while True:
            try:
                self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                return self
            except FileExistsError:
                if time.time() - t0 > self.timeout:
                    raise TimeoutError(f"model registry locked ({self.path}); delete it if no retrain is running")
                time.sleep(0.2)

    def __exit__(self, *a):
        os.close(self.fd)
        os.remove(self.path)


def _ensure():
    os.makedirs(MODELS, exist_ok=True)
    p, cur = os.path.join(MODELS, "v0000.json"), os.path.join(MODELS, "current.json")
    if not os.path.exists(p):
        _write_json(p, Model().to_json())
    if not os.path.exists(cur):
        _write_json(cur, {"version": "v0000"})


def load(version=None) -> Model:
    _ensure()
    version = version or _read_json(os.path.join(MODELS, "current.json"))["version"]
    d = _read_json(os.path.join(MODELS, f"{version}.json"))
    return Model(d["w"], d["bias"], d["version"], d.get("meta"))


def save_new(m: Model, meta: dict) -> Model:
    _ensure()
    with _Lock():
        nums = [int(f[1:5]) for f in os.listdir(MODELS) if f.startswith("v") and f.endswith(".json")]
        m.version = f"v{max(nums) + 1:04d}"
        m.meta = meta
        _write_json(os.path.join(MODELS, f"{m.version}.json"), m.to_json())
    return m


def promote(version: str):
    _ensure()
    if not os.path.exists(os.path.join(MODELS, f"{version}.json")):
        raise FileNotFoundError(f"no model {version}")
    with _Lock():
        _write_json(os.path.join(MODELS, "current.json"), {"version": version})
