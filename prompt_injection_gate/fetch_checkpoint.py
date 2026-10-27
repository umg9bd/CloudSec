"""
fetch_checkpoint.py
===================
Download the Stage-1 classifier checkpoint from the repo's GitHub Release.

The weights (~540 MB) are too large for git (GitHub refuses files over 100 MB),
so `artifacts/` is gitignored and the trained checkpoint ships as a release asset.
checkpoint.json (committed) names the release, the asset and the SHA-256 of
model.safetensors; a download that does not match it is discarded, so everyone
runs the exact model the committed results/<run>.json describes.

The repo is private, so the download needs a GitHub token with read access:
GITHUB_TOKEN in the environment, or `gh auth token` when the GitHub CLI is
logged in. run.cmd / run.sh pass it into the container.

    python prompt_injection_gate/fetch_checkpoint.py            # fetch if missing
    python prompt_injection_gate/fetch_checkpoint.py --force    # replace what is there
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(HERE, "checkpoint.json")
ARTIFACTS = os.path.join(HERE, "artifacts")
API = "https://api.github.com"


def load_manifest() -> dict:
    with open(MANIFEST, encoding="utf-8") as f:
        return json.load(f)


def target_dir(m: dict | None = None) -> str:
    return os.path.join(ARTIFACTS, (m or load_manifest())["dir"])


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def present(m: dict | None = None) -> bool:
    d = target_dir(m)
    return all(os.path.isfile(os.path.join(d, f)) for f in ("gate_model.json", "model.safetensors"))


def token() -> str | None:
    t = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if t:
        return t.strip()
    try:
        out = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else None


def _request(url: str, tok: str, accept: str) -> urllib.request.Request:
    req = urllib.request.Request(url, headers={"Accept": accept, "X-GitHub-Api-Version": "2022-11-28",
                                               "User-Agent": "cloudsec-fetch-checkpoint"})
    # unredirected: the asset URL redirects to signed storage, which must not receive the token
    req.add_unredirected_header("Authorization", f"Bearer {tok}")
    return req


def download(m: dict, tok: str, dest: str) -> None:
    url = f"{API}/repos/{m['repo']}/releases/tags/{m['release_tag']}"
    try:
        with urllib.request.urlopen(_request(url, tok, "application/vnd.github+json"), timeout=30) as r:
            release = json.load(r)
    except urllib.error.HTTPError as e:
        hint = (" (no release with that tag, or this token cannot read the private repo)"
                if e.code == 404 else "")
        raise SystemExit(f"[checkpoint] GitHub API {e.code} for release {m['release_tag']!r}{hint}")
    asset = next((a for a in release.get("assets", []) if a["name"] == m["asset"]), None)
    if asset is None:
        raise SystemExit(f"[checkpoint] release {m['release_tag']!r} has no asset {m['asset']!r}")
    req = _request(f"{API}/repos/{m['repo']}/releases/assets/{asset['id']}", tok, "application/octet-stream")
    total, done, last = asset.get("size") or 0, 0, -1
    with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            pct = int(100 * done / total) if total else 0
            if pct // 10 != last:
                last = pct // 10
                print(f"[checkpoint] {done >> 20} / {total >> 20} MB", flush=True)


def install(zip_path: str, m: dict) -> None:
    """Unpack into a temp dir, check the weights' hash, then move into place."""
    work = tempfile.mkdtemp(prefix=".ckpt-", dir=ARTIFACTS)
    try:
        with zipfile.ZipFile(zip_path) as z:
            root = os.path.realpath(work)
            for name in z.namelist():                     # refuse paths that escape the folder
                if not os.path.realpath(os.path.join(work, name)).startswith(root + os.sep):
                    raise SystemExit(f"[checkpoint] unsafe path in archive: {name!r}")
            z.extractall(work)
        src = os.path.join(work, m["dir"])
        if not os.path.isdir(src):
            raise SystemExit(f"[checkpoint] archive has no top-level folder {m['dir']!r}")
        got = sha256(os.path.join(src, "model.safetensors"))
        if got != m["model_sha256"]:
            raise SystemExit(f"[checkpoint] model.safetensors SHA-256 {got[:12]}... does not match "
                             f"checkpoint.json ({m['model_sha256'][:12]}...); not installed")
        dest = target_dir(m)
        if os.path.exists(dest):
            shutil.rmtree(dest)
        os.replace(src, dest)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def ensure(force: bool = False) -> bool:
    """Fetch the checkpoint unless it is already there. True when it is present afterwards."""
    m = load_manifest()
    if present(m) and not force:
        return True
    tok = token()
    if not tok:
        print(f"[checkpoint] {m['dir']} is missing and no GitHub token is available. Set GITHUB_TOKEN "
              f"(read access to {m['repo']}) or log in with `gh auth login`, then run again.", flush=True)
        return False
    os.makedirs(ARTIFACTS, exist_ok=True)
    print(f"[checkpoint] downloading {m['asset']} from release {m['release_tag']} ...", flush=True)
    fd, zip_path = tempfile.mkstemp(suffix=".zip", dir=ARTIFACTS)
    os.close(fd)
    try:
        download(m, tok, zip_path)
        install(zip_path, m)
    finally:
        os.remove(zip_path)
    print(f"[checkpoint] installed {target_dir(m)} (SHA-256 verified)", flush=True)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="download even if the checkpoint is present")
    args = ap.parse_args()
    sys.exit(0 if ensure(args.force) else 1)


if __name__ == "__main__":
    main()
