@echo off
setlocal enabledelayedexpansion

:: Doc-QnA Demo - Windows Installer
:: Clones repo (if needed), creates venv, installs dependencies

set "INSTALL_DIR=%CD%\doc-qna-openvino"

:: Check Git
where git >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Git is not installed. Please install Git and try again.
    exit /b
)

:: Check Python
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed. Please install Python 3.10-3.12 and try again.
    exit /b
)

:: Clone repository
if not exist "%INSTALL_DIR%" (
    echo Cloning repository...
    git clone https://github.com/bob798/doc-qna-openvino.git "%INSTALL_DIR%"
) else (
    echo Repository already exists. Pulling latest...
    cd /d "%INSTALL_DIR%" && git pull
)

:: Navigate to demo directory
cd /d "%INSTALL_DIR%\openvino"

:: Create virtual environment
echo Creating virtual environment...
python -m venv venv

:: Activate
call venv\Scripts\activate.bat

:: Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

:: Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

:: Set Windows environment variables
set PYTHONIOENCODING=utf-8
set HF_HUB_DISABLE_SYMLINKS=1
set HF_HUB_DISABLE_SYMLINKS_WARNING=1

echo.
echo ========================================
echo All requirements installed for Doc-QnA Demo.
echo You can now run the demo!
echo ========================================
pause
exit
