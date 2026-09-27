@echo off
REM One command for the whole real-time demo: starts the pipeline in Docker and streams a dataset
REM into incoming\ batch by batch, printing every event's HGT / LSTM / risk score and all alerts.
REM Ctrl+C stops everything.
REM
REM   run.cmd                                    real_dataset_test.csv, 200 events every 5 s
REM   run.cmd --feed-interval 2 --feed-limit 2000
REM   run.cmd datasets/privilege-escalation/real_dataset_dev.csv --feed-batch-size 100
cd /d "%~dp0"
docker info >nul 2>&1 || (echo Docker is not running - start Docker Desktop and try again. & exit /b 1)
docker image inspect cloudsec >nul 2>&1 || docker build -t cloudsec . || exit /b 1
docker run --rm -it -v "%cd%:/app" cloudsec python pipeline.py --watch incoming --show-events --reset-state --feed %*
