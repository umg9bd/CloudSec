"""
cloudsec.py -- one command for the whole system.

Starts, and stops together on Ctrl+C:
  1. the Stage 2 hardening watcher      prompt_hardening_realtime/incoming/ -> output/
  2. the real-time pipeline             pipeline.py --watch incoming (HGT + LSTM), printing every
                                        scored event here
  3. the CloudSec app on :8503          cloudsec_app.py, two pages switched from its sidebar:
                                          Agent Gate      ask for an AWS change; it is screened (Stage 1),
                                                          hardened (2), predicted (3), assessed (4) and, if
                                                          approved, executed into incoming/ for (2)
                                          Live Dashboard  what (2) scored: events, alerts, explanations

    run.cmd / ./run.sh                    in Docker (the usual way; builds the image if needed)
    python cloudsec.py                    natively, with an interpreter that has the deps
    python cloudsec.py --keep-state       keep the agent's history from the previous run
    python cloudsec.py --feed             also replay real_dataset_test.csv as background traffic
    python cloudsec.py --feed DATASET --feed-interval 2

Each service's own log is in logs/; this terminal shows the pipeline's live event stream.
"""
from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
import webbrowser

ROOT = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(ROOT, "logs")
STAGE2_ROOT = os.path.join(ROOT, "prompt_hardening_realtime")
APP_PORT = 8503
IN_DOCKER = os.path.exists("/.dockerenv") or os.environ.get("CLOUDSEC_IN_DOCKER") == "1"
REQUIRED = ("torch", "torch_geometric", "pandas", "streamlit", "transformers", "pydantic", "yaml", "jsonschema")


def check_deps() -> None:
    missing = []
    for mod in REQUIRED:
        try:
            __import__(mod)
        except Exception:
            missing.append(mod)
    if missing:
        sys.exit(f"[cloudsec] this interpreter ({sys.executable}) is missing: {', '.join(missing)}.\n"
                 "           Use run.cmd / ./run.sh (Docker), or an environment with the deps "
                 "(e.g. prompt_injection_gate/.venv).")
    if not os.path.isfile(os.path.join(ROOT, "prompt_injection_gate", "artifacts", "distilbert-mbert-pi-v1",
                                       "gate_model.json")):
        print("[cloudsec] WARNING: no Stage 1 classifier checkpoint -- every prompt will be held for review "
              "(fail closed). Train it with: python prompt_injection_gate/run.py --train", flush=True)


def clear_stale_feed_batches() -> int:
    """A stopped demo feed leaves its next batches (<dataset>_batchNNNN.csv) in incoming/; on a fresh
    start they would be scored as if they were new traffic. Only feed batches are removed -- any other
    file someone dropped into incoming/ is still scored."""
    removed = 0
    for folder in (os.path.join(ROOT, "incoming"), os.path.join(ROOT, ".incoming_staging")):
        if os.path.isdir(folder):
            for name in os.listdir(folder):
                if re.fullmatch(r".+_batch\d{4}\.csv", name):
                    os.remove(os.path.join(folder, name))
                    removed += 1
    if removed:
        print(f"[cloudsec] removed {removed} unscored demo-feed batch(es) left in incoming/ by a previous run", flush=True)
    return removed


def start(name: str, cmd: list[str], cwd: str = ROOT, log: bool = True) -> subprocess.Popen:
    os.makedirs(LOG_DIR, exist_ok=True)
    out = open(os.path.join(LOG_DIR, f"{name}.log"), "w", encoding="utf-8") if log else None
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1")
    proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=out, stderr=subprocess.STDOUT if out else None)
    print(f"[cloudsec] started {name}" + (f" (log: logs/{name}.log)" if log else ""), flush=True)
    return proc


def wait_http(url: str, seconds: int = 180) -> bool:
    for _ in range(seconds):
        try:
            urllib.request.urlopen(url, timeout=2)
            return True
        except Exception:
            time.sleep(1)
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keep-state", action="store_true",
                    help="keep per-agent history, scores and alerts from the previous run")
    ap.add_argument("--feed", nargs="?", const="datasets/privilege-escalation/real_dataset_test.csv", metavar="DATASET",
                    help="also replay DATASET into incoming/ as background traffic")
    ap.add_argument("--feed-batch-size", type=int, default=200)
    ap.add_argument("--feed-interval", type=float, default=5.0)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    check_deps()
    # `docker stop` sends SIGTERM: shut the services down the same way Ctrl+C does.
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    if not args.keep_state:
        clear_stale_feed_batches()
    py = sys.executable
    procs = {}
    try:
        procs["stage2-watcher"] = start("stage2-watcher", [py, "-m", "security_hardening", "watch",
                                                           "--workers", "2", "--poll-interval", "0.3"], cwd=STAGE2_ROOT)
        pipe = [py, "pipeline.py", "--watch", "incoming", "--show-events"]
        if not args.keep_state:
            pipe.append("--reset-state")
        if args.feed:
            pipe += ["--feed", args.feed, "--feed-batch-size", str(args.feed_batch_size),
                     "--feed-interval", str(args.feed_interval)]
        procs["pipeline"] = start("pipeline", pipe, log=False)        # its event stream is this terminal
        procs["app"] = start("app", [
            py, "-m", "streamlit", "run", os.path.join(ROOT, "cloudsec_app.py"),
            "--server.address", "0.0.0.0", "--server.port", str(APP_PORT), "--server.headless", "true",
            "--browser.gatherUsageStats", "false", "--theme.base", "dark", "--theme.primaryColor", "#3d63dd",
            "--client.toolbarMode", "viewer"])

        app_url = f"http://localhost:{APP_PORT}"
        if wait_http(f"{app_url}/_stcore/health"):
            print(f"\n[cloudsec] CloudSec: {app_url}  (Agent Gate and Live Dashboard, switched from the sidebar)\n"
                  "[cloudsec] Ctrl+C stops everything.\n", flush=True)
            if not (IN_DOCKER or args.no_browser):
                webbrowser.open(app_url)
        else:
            print("[cloudsec] the CloudSec app did not come up; see logs/app.log", flush=True)

        while True:                                      # stop everything if any service dies
            for name, p in procs.items():
                if p.poll() is not None:
                    print(f"[cloudsec] {name} exited (code {p.returncode}); stopping. "
                          f"See logs/{name}.log", flush=True)
                    return
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        for p in procs.values():
            if p.poll() is None:
                p.terminate()
        for p in procs.values():
            try:
                p.wait(timeout=8)
            except subprocess.TimeoutExpired:
                p.kill()
        print("[cloudsec] stopped.", flush=True)


if __name__ == "__main__":
    main()
