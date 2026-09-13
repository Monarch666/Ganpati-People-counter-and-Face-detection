@echo off
title InsightFace Webcam Test
cd /d "%~dp0"

echo =========================================================
echo              InsightFace Webcam Test
echo =========================================================
echo.
echo Starting InsightFace...
echo (If this is your first run, it will show a download progress bar here)
echo.

python run_insightface.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] The script exited with an error.
)
echo.
pause

