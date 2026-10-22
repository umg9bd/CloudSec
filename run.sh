#!/usr/bin/env bash
# One command for the whole system (macOS/Linux/Git Bash), in Docker -- see run.cmd.
#   the CloudSec app on http://localhost:8503: Agent Gate and Live Dashboard, switched from its sidebar
#   ./run.sh [--keep-state] [--feed [DATASET]]          Ctrl+C stops everything.
set -e
cd "$(dirname "$0")"
docker info >/dev/null 2>&1 || { echo "Docker is not running - start it and try again."; exit 1; }
echo "Building the CloudSec image (first time only takes a few minutes)..."
docker build -q -t cloudsec . >/dev/null
# open the app in the browser once they answer (gives up after 4 min)
( for _ in $(seq 240); do
    if curl -sf -o /dev/null http://localhost:8503/_stcore/health; then
      for url in http://localhost:8503; do
        for opener in xdg-open open explorer.exe; do
          command -v "$opener" >/dev/null && { "$opener" "$url" >/dev/null 2>&1; break; }
        done
        sleep 1
      done
      break
    fi
    sleep 1
  done ) &
# email alerts: the Gmail sender and app password in email.env (see email.env.example), if present
EMAIL_ENV=()
[ -f email.env ] && EMAIL_ENV=(--env-file email.env)
MSYS_NO_PATHCONV=1 docker run --rm -it -p 8503:8503 -v "$PWD:/app" "${EMAIL_ENV[@]}" cloudsec \
  python cloudsec.py "$@"
