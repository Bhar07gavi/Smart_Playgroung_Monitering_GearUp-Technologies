@echo off
REM run.bat
REM Run this every day to start the system
REM Double click it

echo.
echo ================================================
echo    Smart Playground Monitor
echo ================================================
echo.

REM Check venv exists
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: venv not found!
    echo Run setup.bat first
    echo.
    pause
    exit /b 1
)

REM Check .env exists
if not exist ".env" (
    echo ERROR: .env file not found!
    echo Create .env with your Supabase credentials
    echo.
    pause
    exit /b 1
)

REM Activate venv
call venv\Scripts\activate.bat
echo Virtual environment activated
echo.

REM Show Python version
python --version
echo.

REM Test connection first
echo Testing Supabase connection...
python test_connection.py
echo.

REM Ask to continue
set /p START="Start server? (y/n): "
if /i "%START%" neq "y" exit /b 0

echo.
echo Starting Smart Playground Monitor...
echo.
echo Dashboard: http://localhost:8000
echo Press Ctrl+C to stop
echo.

python server.py

pause