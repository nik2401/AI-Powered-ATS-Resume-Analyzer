@echo off
REM ============================================
REM HMM POS Tagger for Hindi - Execution Script
REM ============================================
REM echo Installing dependencies...
REM pip install -r requirements.txt

REM REM Change to script directory
REM cd /d "%~dp0"


REM if not exist "UD_Hindi-HDTB" (
    REM echo Downloading Hindi corpus...
    REM git clone --depth 1 --branch master --single-branch https://github.com/UniversalDependencies/UD_Hindi-HDTB.git UD_Hindi-HDTB
    
REM )




echo [IMPORTANT] Remember to configure paths in Setup.py before running!
echo Open Setup.py and update these variables:
echo   1. HindiCorpus: Path to training corpus
echo   2. HindiCorpustest: Path to test corpus
echo   3. Output.Directory: Where to save results

echo

echo Starting Corpus...

cd /d "C:\Project\ATS_resume-optimizer-ml"

set PYTHON_PATH="C:/Users//nikhi/miniconda3/envs/ats/python.exe"

%PYTHON_PATH% main.py

pause
