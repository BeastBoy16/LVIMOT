@echo off
setlocal EnableExtensions
if not defined CARLA_LAUNCHER set "CARLA_LAUNCHER=C:\CARLA\CarlaUE4.exe"
if not exist "%CARLA_LAUNCHER%" (
    echo [ERROR] CARLA launcher not found: %CARLA_LAUNCHER%
    echo Set CARLA_LAUNCHER to your CarlaUE4.exe path, for example:
    echo   set CARLA_LAUNCHER=D:\CARLA_0.9.13\WindowsNoEditor\CarlaUE4.exe
    exit /b 2
)
for %%I in ("%CARLA_LAUNCHER%") do cd /d "%%~dpI"
"%CARLA_LAUNCHER%" -quality-level=Low -carla-rpc-port=2000 %*
