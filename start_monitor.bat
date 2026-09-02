@echo off
rem Launch monitor (bootstrap; see run_monitor_loop.bat).
rem Pure ASCII on purpose: cmd.exe misreads UTF-8 comments.
setlocal
cd /d "%~dp0"

where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found on PATH. Install Python 3.11+ first.
    pause
    exit /b 1
)

start "monitor" /min cmd /c ""%~dp0run_monitor_loop.bat""
exit /b 0
