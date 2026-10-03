@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "VPY=%CD%\.venv_live\Scripts\python.exe"
if not exist "%VPY%" call setup_live_carla_env.bat
if errorlevel 1 exit /b %errorlevel%
"%VPY%" -u src\carla_main.py --live --evaluate-ground-truth %*
exit /b %ERRORLEVEL%
