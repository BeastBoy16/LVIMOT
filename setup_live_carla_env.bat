@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo ======================================================================
echo LVIMOT Windows environment - CARLA 0.9.13 / Python 3.8
echo ======================================================================

set "PY38="
for /f "delims=" %%P in ('py -3.8 -c "import sys; print(sys.executable)" 2^>nul') do set "PY38=%%P"
if not defined PY38 (
    echo [ERROR] Python 3.8 x64 was not found.
    echo Install Python 3.8 x64 with the Windows py launcher, then rerun this file.
    exit /b 2
)

set "VENV=%CD%\.venv_live"
if not exist "%VENV%\Scripts\python.exe" (
    echo [1/6] Creating .venv_live ...
    py -3.8 -m venv "%VENV%" || exit /b 10
) else (
    echo [1/6] Reusing .venv_live
)
set "VPY=%VENV%\Scripts\python.exe"

echo [2/6] Updating pip tooling ...
"%VPY%" -m pip install --upgrade "pip<25" setuptools wheel || exit /b 11

echo [3/6] Installing LVIMOT dependencies ...
"%VPY%" -m pip install --upgrade --force-reinstall -r requirements-windows.txt || exit /b 12

echo [4/6] Installing CUDA PyTorch ...
"%VPY%" -m pip install --upgrade --force-reinstall torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121 || exit /b 13

echo [5/6] Verifying packages ...
"%VPY%" -c "import sys, carla, cv2, numpy, scipy, torch, PIL; print('Python:',sys.version); print('CARLA module:',carla.__file__); print('OpenCV:',cv2.__version__); print('Torch:',torch.__version__); print('CUDA:',torch.cuda.is_available())" || exit /b 14

echo [6/6] Ready.
echo Run: run_live_lvimot.bat
exit /b 0
