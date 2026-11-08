"""Deploy the public demo to a Hugging Face Space (Docker SDK, free CPU).

    HF_TOKEN=hf_... python deploy/push_space.py [space-name]

What it does:
1. creates the Space <your-user>/<space-name> (default promo-finder) if it doesn't exist, public, Docker SDK;
2. stores JUDGE_API_KEY from your local .env as a Space SECRET (never committed, never printed);
3. pushes only what the demo needs: Dockerfile, the promofinder package, models, gold labels, the fictional sample,
   and the Space README. The confidential scripts, run outputs and .env are never copied.
The Space builds the container itself; it is live at https://huggingface.co/spaces/<user>/<space-name> in ~2-4 min.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API = "https://huggingface.co/api"


def call(method, path, token, body=None):
    req = urllib.request.Request(API + path, None if body is None else json.dumps(body).encode(), method=method,
                                 headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read()
            return r.status, (json.loads(raw) if raw else {})
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode(errors="replace")[:300]}


def env_value(key):
    if os.environ.get(key):
        return os.environ[key]
    path = os.path.join(ROOT, ".env")
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            k, _, v = line.strip().partition("=")
            if k == key and v:
                return v.split(" #", 1)[0].strip()
    return None


def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("Set HF_TOKEN (a Hugging Face token with write access).")
    name = sys.argv[1] if len(sys.argv) > 1 else "promo-finder"
    code, me = call("GET", "/whoami-v2", token)
    if code != 200:
        sys.exit(f"Token rejected by Hugging Face ({code}).")
    repo = f"{me['name']}/{name}"

    code, res = call("POST", "/repos/create", token, {"type": "space", "name": name, "sdk": "docker", "private": False})
    if code not in (200, 201, 409):
        sys.exit(f"Couldn't create the Space ({code}): {res}")
    print("space:", repo, "(created)" if code != 409 else "(exists)")

    key = env_value("JUDGE_API_KEY")
    if key:
        code, res = call("POST", f"/spaces/{repo}/secrets", token, {"key": "JUDGE_API_KEY", "value": key})
        print("secret JUDGE_API_KEY:", "set" if code in (200, 201) else f"failed ({code})")
    else:
        print("no JUDGE_API_KEY found: the demo will offer offline runs only")

    with tempfile.TemporaryDirectory() as tmp:
        for f in ("Dockerfile", ".dockerignore"):
            shutil.copy(os.path.join(ROOT, f), tmp)
        shutil.copy(os.path.join(ROOT, "deploy", "hf", "README.md"), os.path.join(tmp, "README.md"))
        ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "*.tmp*", ".lock")
        for d in ("promofinder", "gold", "models", "samples"):
            shutil.copytree(os.path.join(ROOT, d), os.path.join(tmp, d), ignore=ignore)
        for bad in (".env", "data", "runs"):  # belt and braces
            assert not os.path.exists(os.path.join(tmp, bad)), bad
        git = lambda *a: subprocess.run(["git", *a], cwd=tmp, check=True, capture_output=True, text=True)
        git("init", "-q", "-b", "main")
        git("-c", "user.name=deploy", "-c", "user.email=deploy@localhost", "add", "-A")
        git("-c", "user.name=deploy", "-c", "user.email=deploy@localhost", "commit", "-q", "-m", "Deploy Promo Finder demo")
        url = f"https://user:{token}@huggingface.co/spaces/{repo}"
        r = subprocess.run(["git", "push", "-q", "--force", url, "main"], cwd=tmp, capture_output=True, text=True)
        if r.returncode:
            sys.exit("push failed: " + r.stderr.replace(token, "***")[:400])
    print(f"pushed. Building now; live in ~2-4 minutes at https://huggingface.co/spaces/{repo}")
    print(f"direct app URL: https://{repo.replace('/', '-').lower()}.hf.space")


if __name__ == "__main__":
    main()
