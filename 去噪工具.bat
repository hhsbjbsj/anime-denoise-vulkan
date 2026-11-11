@echo off
rem Opens the denoise web app with the first Python that can import its dependencies:
rem DENOISE_PYTHON (full path to a python.exe), .venv next to this file, H:\ComfyUI-latest\.venv,
rem then the py launcher and python on PATH. When none can, setup_env.py installs them into .venv
rem in this window first. Keep this file ASCII: cmd misreads UTF-8 batch files.
rem
rem On a computer with an NVIDIA driver (nvidia-smi on PATH) a Python whose torch is a CPU-only build
rem is skipped, so a CPU torch in a global Python no longer hides the one-click CUDA install.
rem DENOISE_PYTHON is always used as given, CPU build or not.
setlocal
set "PYEXE="
set "DENOISE_NEED_GPU="
where nvidia-smi >nul 2>nul && set "DENOISE_NEED_GPU=1"
call :find
if defined PYEXE goto launch

set "BASEPY="
where py >nul 2>nul && set "BASEPY=py -3"
if not defined BASEPY where python >nul 2>nul && set "BASEPY=python"
if not defined BASEPY goto nopython
%BASEPY% "%~dp0setup_env.py" || goto failed
rem Accept what setup installed even when it had to fall back to the CPU build; it says why.
set "DENOISE_NEED_GPU="
call :find
if not defined PYEXE goto failed

:launch
rem pythonw keeps the app running without a console window
set "PYW=%PYEXE:python.exe=pythonw.exe%"
if not exist "%PYW%" set "PYW=%PYEXE%"
start "" "%PYW%" "%~dp0denoise_app.py"
exit /b 0

:find
if defined DENOISE_PYTHON call :tryany "%DENOISE_PYTHON%"
if not defined PYEXE call :try "%~dp0.venv\Scripts\python.exe"
if not defined PYEXE call :try "H:\ComfyUI-latest\.venv\Scripts\python.exe"
if not defined PYEXE call :try py -3
if not defined PYEXE call :try python
exit /b 0

:try
rem Records the interpreter's real path only when all the imports succeed and, when DENOISE_NEED_GPU
rem is set, torch is a CUDA or ROCm build.
for /f "usebackq delims=" %%P in (`call %* -c "import os, sys, torch, spandrel, numpy, PIL; sys.exit(1) if os.environ.get('DENOISE_NEED_GPU') and not (torch.version.cuda or getattr(torch.version, 'hip', None)) else print(sys.executable)" 2^>nul`) do set "PYEXE=%%P"
exit /b 0

:tryany
for /f "usebackq delims=" %%P in (`call %* -c "import torch, spandrel, numpy, PIL, sys; print(sys.executable)" 2^>nul`) do set "PYEXE=%%P"
exit /b 0

:nopython
echo Python was not found. Install Python 3.10 or newer from https://www.python.org/downloads/
echo and tick "Add python.exe to PATH", then double-click this file again.
pause
exit /b 1

:failed
echo.
echo Setup did not finish, see the messages above.
pause
exit /b 1
