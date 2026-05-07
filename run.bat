@echo off
title Deepfake Detector Setup and Run
echo =========================================
echo    Deepfake Detector Startup Script
echo =========================================
echo.

IF NOT EXIST "venv" (
    echo [INFO] Creating virtual environment (this only happens once)...
    python -m venv venv
)

echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat

echo [INFO] Checking dependencies...
pip install -r requirements.txt --quiet

echo [INFO] Starting the application...
echo.
python app.py

pause
