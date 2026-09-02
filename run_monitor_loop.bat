@echo off
rem monitor auto-restart loop: restart 5s after any crash.
rem Pure ASCII on purpose: cmd.exe misreads UTF-8 comments.
setlocal
cd /d "%~dp0"

where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found on PATH.
    pause
    exit /b 1
)

:loop
echo [%date% %time%] starting monitor ...
python -m monitor.main
echo [%date% %time%] monitor exited (%ERRORLEVEL%); restarting in 5s ...
timeout /t 5 /nobreak >nul
goto loop
