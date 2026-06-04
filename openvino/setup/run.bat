@echo off
setlocal enabledelayedexpansion

:: Doc-QnA Demo - Windows Runner
set "INSTALL_DIR=%CD%\doc-qna-openvino\openvino"

cd /d "%INSTALL_DIR%"

:: Check venv
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found! Please run install.bat first.
    exit /b
)

:: Activate
echo Activating virtual environment...
call venv\Scripts\activate.bat

:: Set environment
set PYTHONIOENCODING=utf-8
set HF_HUB_DISABLE_SYMLINKS=1
set HF_HUB_DISABLE_SYMLINKS_WARNING=1

:: Run demo
echo Running Doc-QnA Demo...
python main.py

pause
exit
