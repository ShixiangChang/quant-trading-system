@echo off
rem Launch pin-strategy paper engine (one-click).
rem Event-driven paper trading: -5% 15min drop -> long -> hold 12h.
rem Pure ASCII on purpose: cmd.exe misreads UTF-8 comments.
setlocal
cd /d "%~dp0"

where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found on PATH.
    pause
    exit /b 1
)

start "pin" /min cmd /c "python -m paper.pin_engine --loop"
exit /b 0
