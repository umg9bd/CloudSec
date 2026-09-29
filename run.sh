#!/usr/bin/env bash
# One command for everything (macOS/Linux/Git Bash): pipeline + demo feed + dashboard on
# http://localhost:8501 -- see run.cmd. Ctrl+C stops everything.
set -e
cd "$(dirname "$0")"
docker info >/dev/null 2>&1 || { echo "Docker is not running - start it and try again."; exit 1; }
docker build -q -t cloudsec . >/dev/null
# open the dashboard in the browser once it answers (gives up after 3 min)
( for _ in $(seq 180); do
    if curl -sf -o /dev/null http://localhost:8501/_stcore/health; then
      for opener in xdg-open open explorer.exe; do
        command -v "$opener" >/dev/null && { "$opener" http://localhost:8501 >/dev/null 2>&1; break; }
      done
      break
    fi
    sleep 1
  done ) &
MSYS_NO_PATHCONV=1 docker run --rm -it -p 8501:8501 -v "$PWD:/app" cloudsec \
  python pipeline.py --watch incoming --show-events --reset-state --dashboard --feed "$@"
