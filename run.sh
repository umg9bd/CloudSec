#!/usr/bin/env bash
# One command for the whole real-time demo (macOS/Linux/Git Bash): see run.cmd.
set -e
cd "$(dirname "$0")"
docker info >/dev/null 2>&1 || { echo "Docker is not running - start it and try again."; exit 1; }
docker image inspect cloudsec >/dev/null 2>&1 || docker build -t cloudsec .
MSYS_NO_PATHCONV=1 docker run --rm -it -v "$PWD:/app" cloudsec \
  python pipeline.py --watch incoming --show-events --reset-state --feed "$@"
