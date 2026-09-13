@echo off
title People Counter — Web Dashboard
cd /d "%~dp0"

echo ======================================================================
echo          People Counter — Web Dashboard
echo ======================================================================
echo.
echo Camera IP:    192.168.1.35
echo Dashboard:    http://localhost:5002
echo.
echo Starting server...
echo (Open your browser to http://localhost:5002)
echo.

python web_dashboard.py --ip 192.168.1.35 --user admin --password 123456 --port 5002 %*

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Dashboard exited with an error.
    pause
)

