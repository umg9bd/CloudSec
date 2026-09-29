@echo off
REM One command for everything: builds the Docker image if needed, clears the previous run's
REM output, starts the pipeline, streams a dataset into incoming\ batch by batch (printing every
REM event's HGT / LSTM / risk score and all alerts) and serves the dashboard on
REM http://localhost:8501, which opens in the browser. Ctrl+C stops everything.
REM
REM   run.cmd                                    real_dataset_test.csv, 200 events every 5 s
REM   run.cmd --feed-interval 2 --feed-limit 2000
REM   run.cmd datasets/privilege-escalation/real_dataset_dev.csv --feed-batch-size 100
cd /d "%~dp0"
docker info >nul 2>&1 || (echo Docker is not running - start Docker Desktop and try again. & exit /b 1)
docker build -q -t cloudsec . >nul || exit /b 1
REM Opens the dashboard in the browser once it answers (hidden window; gives up after 3 min).
start "" /min powershell -NoProfile -WindowStyle Hidden -Command "for ($i = 0; $i -lt 180; $i++) { try { Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 http://localhost:8501/_stcore/health | Out-Null; Start-Process http://localhost:8501; break } catch { Start-Sleep 1 } }"
docker run --rm -it -p 8501:8501 -v "%cd%:/app" cloudsec python pipeline.py --watch incoming --show-events --reset-state --dashboard --feed %*
