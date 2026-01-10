@echo off
echo ========================================
echo MedivisionAI Backend Server
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

echo [1/3] Checking dependencies...
pip show fastapi >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    pip install -r requirements.txt
) else (
    echo Dependencies already installed
)

echo.
echo [2/3] Checking model files...
if not exist "stageA_10.pth" (
    echo WARNING: stageA_10.pth not found
)
if not exist "stageB_10.pth" (
    echo WARNING: stageB_10.pth not found
)
if not exist "body_model.pth" (
    echo WARNING: body_model.pth not found
)
if not exist "fracture_model.pth" (
    echo WARNING: fracture_model.pth not found
)

echo.
echo [3/3] Starting server...
echo Server will be available at: http://localhost:8000
echo API endpoint: http://localhost:8000/api/analyze
echo.
echo Press Ctrl+C to stop the server
echo ========================================
echo.

python main.py
