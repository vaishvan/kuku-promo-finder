"""Stage 3b: rubric judge. One interface, two backends.

judge(view) -> {hook, cliffhanger, standalone, emotion: 0-10, reason, by}
- OfflineJudge: deterministic lexicon stand-in. Keeps the system runnable/testable with no key.
- LLMJudge: any OpenAI-compatible chat endpoint (Gemini/Groq/OpenRouter/Ollama free tiers all expose one).
  Set JUDGE_BASE_URL, JUDGE_API_KEY, JUDGE_MODEL. Results are cached by content hash.
"""
import hashlib
import json
import os
import random
import re
import time
import urllib.error
import urllib.request

from .features import AROUSAL_TAGS, CONFLICT, HOOK_ACTION

RUBRIC = """You grade candidate promo clips cut from a microdrama screenplay (timestamps are seconds in the episode).
A good promo: (1) HOOK - the first 3 seconds make a scrolling viewer stop (a slap, reveal, shocking line);
(2) CLIFFHANGER - it ends unresolved so the viewer must know what happens next;
(3) STANDALONE - someone who has never seen the show understands who is fighting and why.
Score each 0-10 (be harsh; 5 is average) plus EMOTION 0-10 (high-arousal: shock, betrayal, fear, rage).
Reply with JSON only: {"hook":n,"cliffhanger":n,"standalone":n,"emotion":n,"reason":"<one line, max 25 words>"}"""


def render(view) -> str:
    return "\n".join(view["lines"])[:6000]


class OfflineJudge:
    name = "offline-lexicon"

    def __call__(self, view):
        text = " ".join(view["lines"])
        first, last = view["hook_line"] or "", view["cliff_line"] or ""
        hook = 2 + 2 * ("!" in first) + 1 * ("?" in first) + 2 * bool(CONFLICT.search(first)) + 2 * bool(HOOK_ACTION.search(first)) + 1 * (len(first) < 80)
        cliff = 2 + 3 * bool(last.rstrip('"').endswith(("...", "?", "—"))) + 3 * bool(CONFLICT.search(last)) + 1 * ("!" in last)
        stand = 8 - 2 * len(re.findall(r"\bas i said|earlier|last time\b", text, re.I)) - (1 if len(set(re.findall(r"\b[A-Z]{3,}:", text))) > 3 else 0)
        emo = 2 + min(6, len(AROUSAL_TAGS.findall(text)) // 2) + 1 * ("!" in text)
        c = lambda x: max(0, min(10, x))
        return {"hook": c(hook), "cliffhanger": c(cliff), "standalone": c(stand), "emotion": c(emo),
                "reason": "offline lexicon heuristic", "by": self.name}


PAIR_RUBRIC = """You are choosing which of two candidate promo clips (cut from the same microdrama) to post as an ad.
Judge by the same three tests: hook in the first 3 seconds, ends unresolved, understandable to a new viewer.
Reply with JSON only: {"better":"A" or "B","confidence":0.5-1.0}"""
RUBRIC_HASH = hashlib.sha256((RUBRIC + PAIR_RUBRIC).encode()).hexdigest()[:12]  # cache key follows the text (fix D4)
SCORE_KEYS = ("hook", "cliffhanger", "standalone", "emotion")


def first_json(text):
    """First complete JSON object in a reply (a greedy regex broke on prose containing braces, fix D5)."""
    dec = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch == "{":
            try:
                obj, _ = dec.raw_decode(text[i:])
                if isinstance(obj, dict):
                    return obj
            except ValueError:
                continue
    raise ValueError("no JSON object in reply")


class BudgetExceeded(RuntimeError):
    pass


class LLMJudge:
    RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}

    def __init__(self, cache_path="runs/judge_cache.jsonl", attempts=4, base_delay=2.0, sleep=None):
        self.base = os.environ["JUDGE_BASE_URL"].rstrip("/")
        self.key = os.environ["JUDGE_API_KEY"]
        self.model = os.environ["JUDGE_MODEL"]
        self.name = f"llm:{self.model}"
        self.cache_path = cache_path
        self.attempts, self.base_delay = attempts, base_delay
        self.sleep = sleep or time.sleep
        self.stats = {"calls": 0, "cache_hits": 0, "retries": 0, "failures": 0, "api_seconds": 0.0, "new": 0, "over_budget": 0}
        self.max_new = None  # cap on uncached judgements (free-tier quota); cached ones are always served
        self.cache = {}
        if os.path.exists(cache_path):
            with open(cache_path, encoding="utf-8") as f:
                for l in f:
                    r = json.loads(l)
                    self.cache[r["k"]] = r["v"]
        self.fallback = OfflineJudge()

    def _post(self, system, text):
        payload = {"model": self.model, "temperature": 0,
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": text}]}
        if os.environ.get("JUDGE_JSON_MODE", "1") == "1":
            payload["response_format"] = {"type": "json_object"}
        body = json.dumps(payload).encode()
        req = urllib.request.Request(self.base + "/chat/completions", body,
                                     {"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.load(r)["choices"][0]["message"]["content"]

    def _ask(self, system, text, parse):
        """Exponential backoff with jitter on rate limits / server errors / timeouts / unparsable replies (fix D2)."""
        last = None
        for i in range(self.attempts):
            t0 = time.time()
            try:
                self.stats["calls"] += 1
                out = parse(first_json(self._post(system, text)))
                self.stats["api_seconds"] += time.time() - t0
                return out
            except urllib.error.HTTPError as ex:
                last = ex
                if ex.code not in self.RETRY_STATUS:
                    break
            except (urllib.error.URLError, TimeoutError, ValueError, KeyError, TypeError) as ex:
                last = ex
            if i + 1 < self.attempts:
                self.stats["retries"] += 1
                self.sleep(self.base_delay * (2 ** i) * (1 + random.random() * 0.25))
        self.stats["failures"] += 1
        raise last or RuntimeError("judge failed")

    def _cached(self, key, fn):
        if key in self.cache:
            self.stats["cache_hits"] += 1
            return self.cache[key]
        if self.max_new is not None and self.stats["new"] >= self.max_new:
            self.stats["over_budget"] += 1
            raise BudgetExceeded(f"free-request budget of {self.max_new} new calls used up")
        self.stats["new"] += 1
        v = fn()
        os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
        with open(self.cache_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"k": key, "v": v}) + "\n")
        self.cache[key] = v
        return v

    def __call__(self, view):
        text = render(view)
        k = hashlib.sha256(f"{RUBRIC_HASH}|{self.model}|{text}".encode()).hexdigest()

        def parse(d):
            v = {x: float(d[x]) for x in SCORE_KEYS}
            if any(not 0 <= s <= 10 for s in v.values()):
                raise ValueError(f"score out of range: {v}")
            v.update(reason=str(d.get("reason", ""))[:200], by=self.name)
            return v
        try:
            return self._cached(k, lambda: self._ask(RUBRIC, text, parse))
        except Exception as ex:  # flagged in `by`; featurize refuses runs with too many of these
            v = self.fallback(view)
            v["by"] = f"offline-fallback({type(ex).__name__})"
            return v  # not cached

    def compare(self, view_a, view_b):
        """P(clip A is the better promo). Call twice with the order swapped to cancel position bias (fix D3)."""
        text = f"CLIP A\n{render(view_a)[:3000]}\n\nCLIP B\n{render(view_b)[:3000]}"
        k = hashlib.sha256(f"pair|{RUBRIC_HASH}|{self.model}|{text}".encode()).hexdigest()

        def parse(d):
            b, c = str(d["better"]).strip().upper(), min(1.0, max(0.5, float(d.get("confidence", 0.75))))
            if b not in ("A", "B"):
                raise ValueError(f"bad choice {b!r}")
            return c if b == "A" else 1 - c
        try:
            return self._cached(k, lambda: self._ask(PAIR_RUBRIC, text, parse))
        except Exception:
            return 0.5  # no information


def load_env(path=".env"):
    """Minimal .env reader (KEY=VALUE lines). Never overrides variables already set; the file is gitignored."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                if v.strip() and not os.environ.get(k.strip()):
                    os.environ[k.strip()] = v.strip()


def get_judge(offline=False):
    load_env()
    if offline or not (os.environ.get("JUDGE_API_KEY") and os.environ.get("JUDGE_MODEL")):
        return OfflineJudge()
    return LLMJudge(attempts=int(os.environ.get("JUDGE_ATTEMPTS", 2)), base_delay=float(os.environ.get("JUDGE_BACKOFF", 5)))
