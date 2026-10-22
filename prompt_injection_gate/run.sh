#!/usr/bin/env bash
# One command to set up and launch the Stage-1 prompt-screening gate (macOS/Linux/Git Bash).
#   ./run.sh              set up if needed, then open the UI
#   ./run.sh --no-train   launch even without a trained model
#   ./run.sh --train      force a fresh training run first
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python "$DIR/run.py" "$@"
