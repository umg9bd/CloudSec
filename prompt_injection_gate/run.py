"""
run.py -- one command to set up and launch the Stage-1 prompt-screening gate.

Does, in order, each step only if it is needed:
  1. create the module venv (.venv, system-site-packages) if missing;
  2. install the module deps into it (transformers, datasets, streamlit, pyyaml)
     if they are not importable there;
  3. build the merged dataset (merge_public.py) if it is missing;
  4. train the classifier (train_eval_transformer.py) if no checkpoint exists
     -- skip with --no-train (the gate then fails closed to REVIEW), force with --train;
  5. launch the Streamlit UI.

Runs with the BASE interpreter (stdlib only); it creates and then uses the venv.

    python prompt_injection_gate/run.py              # set up if needed, then launch the UI
    python prompt_injection_gate/run.py --no-train   # launch even without a trained model
    python prompt_injection_gate/run.py --train      # force a fresh training run first
    python prompt_injection_gate/run.py --port 8600
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
IS_WIN = sys.platform == "win32"
VENV = os.path.join(HERE, ".venv")
VPY = os.path.join(VENV, "Scripts", "python.exe") if IS_WIN else os.path.join(VENV, "bin", "python")
DEPS = ["transformers>=4.50", "datasets>=3", "streamlit==1.64.0", "pyyaml>=6", "sentencepiece", "protobuf",
        "pydantic>=2"]   # pydantic: Stage 3 (log-generation) schemas, used by the downstream chain
DATASET = os.path.join(ROOT, "datasets", "prompt-injection")
MERGED = os.path.join(DATASET, "prompt_injection_merged.jsonl")
sys.path.insert(0, ROOT)
from prompt_injection_gate.config import load_config  # noqa: E402
CKPT = load_config().classifier_dir


def step(msg):
    print(f"\n\033[1m[gate] {msg}\033[0m", flush=True)


def run(cmd, **kw):
    print("  $ " + " ".join(os.path.basename(c) if c.endswith("python.exe") else c for c in cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


def ensure_venv():
    if os.path.isfile(VPY):
        return
    step("creating module venv (.venv)")
    run([sys.executable, "-m", "venv", "--system-site-packages", VENV])


def deps_present():
    code = "import transformers, datasets, streamlit, yaml, torch, pydantic"  # torch comes from system-site
    return subprocess.run([VPY, "-c", code], capture_output=True).returncode == 0


def ensure_deps():
    if deps_present():
        return
    step("installing module dependencies into the venv (first run only)")
    run([VPY, "-m", "pip", "install", "--disable-pip-version-check", "-q", *DEPS])


def ensure_dataset():
    if os.path.isfile(MERGED):
        return
    step("building the merged dataset (public sets + cloud slice)")
    run([VPY, os.path.join(DATASET, "merge_public.py")], cwd=ROOT)


def ensure_model(mode):
    have = os.path.isfile(os.path.join(CKPT, "gate_model.json")) and \
        os.path.isfile(os.path.join(CKPT, "model.safetensors"))
    if mode == "no-train":
        if not have:
            print("  (no checkpoint; the gate will fail closed to REVIEW_REQUIRED)")
        return
    if have and mode != "force":
        return
    step("training the classifier (CPU, ~15-20 min; one-time)")
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    run([VPY, "-u", os.path.join(HERE, "training", "train_eval_transformer.py")], cwd=ROOT, env=env)


def ensure_streamlit_creds():
    """Pre-answer Streamlit's one-time email prompt (empty), so a non-interactive
    launch does not hang/exit on it. Same file Streamlit writes if you answer it."""
    creds = os.path.join(os.path.expanduser("~"), ".streamlit", "credentials.toml")
    if os.path.isfile(creds):
        return
    os.makedirs(os.path.dirname(creds), exist_ok=True)
    with open(creds, "w", encoding="utf-8") as f:
        f.write('[general]\nemail = ""\n')


def launch_ui(port):
    ensure_streamlit_creds()
    step(f"launching the UI at http://localhost:{port}  (Ctrl+C to stop)")
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    run([VPY, "-m", "streamlit", "run", os.path.join(HERE, "ui_app.py"),
         "--server.port", str(port), "--server.headless", "false",
         "--browser.gatherUsageStats", "false"], cwd=ROOT, env=env)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--train", action="store_true", help="force a fresh training run first")
    g.add_argument("--no-train", action="store_true", help="launch even without a trained model")
    ap.add_argument("--port", type=int, default=8502)
    args = ap.parse_args()

    ensure_venv()
    ensure_deps()
    ensure_dataset()
    ensure_model("force" if args.train else "no-train" if args.no_train else "auto")
    launch_ui(args.port)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(f"\n[gate] step failed (exit {e.returncode}). See the message above.", file=sys.stderr)
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\n[gate] stopped.")
