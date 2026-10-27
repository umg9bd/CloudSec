@echo off
REM One command for the whole system (cloudsec.py), in Docker. Builds the image if needed, then starts:
REM   the CloudSec app    http://localhost:8503  -- sidebar switches between its two pages:
REM     Agent Gate:     ask for an AWS change; it is screened (Stage 1),
REM                       hardened (Stage 2, via its incoming/ folder), predicted (Stage 3), risk-scored
REM                       by HGT + LSTM (Stage 4), and, if approved, executed (simulated) into incoming/
REM     Live Dashboard: what the real-time pipeline scored (this window also shows every event)
REM It opens in the browser. Ctrl+C stops everything.
REM
REM   run.cmd                        fresh start, agent traffic only
REM   run.cmd --keep-state           keep the agent's history, scores and alerts from the last run
REM   run.cmd --feed                 also replay real_dataset_test.csv as background traffic
REM   run.cmd --feed datasets/privilege-escalation/real_dataset_dev.csv --feed-interval 2
cd /d "%~dp0"
docker info >nul 2>&1 || (echo Docker is not running - start Docker Desktop and try again. & exit /b 1)
echo Building the CloudSec image (first time only takes a few minutes)...
docker build -q -t cloudsec . >nul || exit /b 1
REM Opens the app once it answers (hidden window; gives up after 4 min).
start "" /min powershell -NoProfile -WindowStyle Hidden -Command "for ($i = 0; $i -lt 240; $i++) { try { Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 http://localhost:8503/_stcore/health | Out-Null; Start-Process http://localhost:8503; break } catch { Start-Sleep 1 } }"
REM Email alerts: the Gmail sender and app password in email.env (see email.env.example), if present.
set "EMAIL_ENV="
if exist email.env set "EMAIL_ENV=--env-file email.env"
REM Stage 1 classifier: if it is not in prompt_injection_gate\artifacts yet, cloudsec.py downloads it from the
REM repo's GitHub Release, which needs read access: GITHUB_TOKEN, or a logged-in GitHub CLI (gh auth login).
if not defined GITHUB_TOKEN for /f "delims=" %%t in ('gh auth token 2^>nul') do set "GITHUB_TOKEN=%%t"
docker run --rm -it -p 8503:8503 -v "%cd%:/app" -e GITHUB_TOKEN %EMAIL_ENV% cloudsec python cloudsec.py %*
