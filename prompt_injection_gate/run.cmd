@echo off
REM One command to set up and launch the Stage-1 prompt-screening gate (Windows).
REM   run.cmd              set up if needed, then open the UI
REM   run.cmd --no-train   launch even without a trained model
REM   run.cmd --train      force a fresh training run first
python "%~dp0run.py" %*
