@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set PYTHONFAULTHANDLER=1
set PYTHONUNBUFFERED=1
set "VPY=%CD%\.venv_live\Scripts\python.exe"
if not exist "%VPY%" (
    call setup_live_carla_env.bat
    if errorlevel 1 exit /b %errorlevel%
)
"%VPY%" -u src\lvimot_live_gui.py
exit /b %ERRORLEVEL%
