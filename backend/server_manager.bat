@echo off
REM MedivisionAI Backend Server Manager

:menu
cls
echo ========================================
echo MedivisionAI Backend Server Manager
echo ========================================
echo.
echo 1. Start Server
echo 2. Stop Server (if running)
echo 3. Restart Server
echo 4. Test API
echo 5. Check Server Status
echo 6. Exit
echo.
set /p choice="Enter your choice (1-6): "

if "%choice%"=="1" goto start
if "%choice%"=="2" goto stop
if "%choice%"=="3" goto restart
if "%choice%"=="4" goto test
if "%choice%"=="5" goto status
if "%choice%"=="6" goto end
goto menu

:start
echo.
echo Starting MedivisionAI Backend Server...
echo.
start "MedivisionAI Backend" python main.py
timeout /t 5 /nobreak >nul
echo.
echo Server started! Running at http://localhost:8000
echo Check the new window for server logs.
echo.
pause
goto menu

:stop
echo.
echo Stopping server on port 8000...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000') do (
    taskkill /PID %%a /F 2>nul
)
echo Server stopped.
echo.
pause
goto menu

:restart
echo.
echo Restarting server...
echo.
echo Step 1: Stopping existing server...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000') do (
    taskkill /PID %%a /F 2>nul
)
timeout /t 2 /nobreak >nul
echo.
echo Step 2: Starting new server...
start "MedivisionAI Backend" python main.py
timeout /t 5 /nobreak >nul
echo.
echo Server restarted! Running at http://localhost:8000
echo.
pause
goto menu

:test
echo.
echo Running API tests...
echo.
python api_test.py
echo.
pause
goto menu

:status
echo.
echo Checking server status...
echo.
netstat -ano | findstr :8000
if errorlevel 1 (
    echo Server is NOT running on port 8000
) else (
    echo Server is RUNNING on port 8000
    echo.
    echo Testing health endpoint...
    curl http://localhost:8000/ 2>nul
)
echo.
pause
goto menu

:end
echo.
echo Goodbye!
timeout /t 2 /nobreak >nul
exit
