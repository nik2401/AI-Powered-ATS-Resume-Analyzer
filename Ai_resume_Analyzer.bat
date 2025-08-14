@echo off
REM ============================================
REM AI Powered ATS Analyzer - Execution Script
REM ============================================



echo [IMPORTANT] Remember to configure paths in Setup.py before running!
echo Open Setup.py and update these variables:

echo

echo Starting Analyzer...

cd /d "Project/ATS_resume-optimizer-ml"

set PYTHON_PATH="python exe path"  REM Adjust if Python is not in PATH

%PYTHON_PATH% main.py

pause
