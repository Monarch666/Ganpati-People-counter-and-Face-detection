@echo off
title Zone-Based Queue Counter (Dense Crowd)
cd /d "%~dp0"

echo ======================================================================
echo          Starting Zone-Based Queue Counter (YOLOv8s)
echo ======================================================================
echo.

:: Check if python is accessible
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python was not found in your PATH.
    echo Please install Python 3.10+ and make sure 'Add Python to PATH' is checked.
    echo.
    pause
    exit /b 1
)

:: Run counter.py with any extra arguments passed, defaulting to webcam index 0
echo Starting webcam counter...
echo.
echo === SETUP ===
echo On first run, a window will pop up. 
echo 1. Left-click to draw your queue zone.
echo 2. Right-click or press Enter when done.
echo (The zone saves automatically for future runs)
echo.
echo === HOTKEYS ===
echo [Q] / [ESC] - Quit
echo [R] - Reset counts
echo [D] - Redraw the queue zone
echo [T] - Toggle trajectory trails
echo.

python counter.py %*

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [NOTE] Application exited with code %ERRORLEVEL%.
    pause
)
