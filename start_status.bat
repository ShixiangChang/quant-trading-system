@echo off
rem Launch status dashboard (one-click): start server + open browser.
rem Pure ASCII on purpose: cmd.exe misreads UTF-8 comments.
setlocal
cd /d "%~dp0"

set "PORT=8777"

where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found on PATH.
    pause
    exit /b 1
)

rem ---- kill any process still holding the port ----
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":%PORT%" ^| findstr "LISTENING"') do (
    taskkill /F /PID %%a >nul 2>&1
)

rem ---- start server in a minimized window ----
start "status" /min python status.py serve --port %PORT%

rem ---- wait for server, then open browser ----
timeout /t 2 /nobreak >nul
start "" http://127.0.0.1:%PORT%

exit /b 0
