@echo off
REM setup.bat
REM Run this file once to setup everything
REM Double click it OR run in CMD

echo.
echo ================================================
echo    Smart Playground Monitor - Setup
echo    Python 3.11
echo ================================================
echo.

REM Use exact Python 3.11 path
set PYTHON=C:\Users\Bhargavi\AppData\Local\Programs\Python\Python311\python.exe

REM Check Python exists
if not exist "%PYTHON%" (
    echo ERROR: Python not found at:
    echo %PYTHON%
    echo.
    echo Install Python 3.11 from python.org
    pause
    exit /b 1
)

echo Python found: %PYTHON%
echo.

REM Create venv
echo Creating virtual environment...
"%PYTHON%" -m venv venv

if errorlevel 1 (
    echo ERROR: Failed to create venv!
    pause
    exit /b 1
)

echo venv created OK
echo.

REM Activate
call venv\Scripts\activate.bat
echo venv activated
echo.

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip --quiet
echo pip upgraded
echo.

REM Install packages
echo Installing packages for Python 3.11...
echo This takes 5-15 minutes, please wait...
echo.

echo [1/7] Installing basic packages...
pip install python-dotenv requests Pillow --quiet

echo [2/7] Installing web server...
pip install fastapi uvicorn websockets aiofiles python-multipart --quiet

echo [3/7] Installing OpenCV...
pip install opencv-python==4.8.1.78 --quiet

echo [4/7] Installing NumPy...
pip install numpy==1.24.3 --quiet

echo [5/7] Installing Supabase...
pip install supabase==2.3.4 --quiet

echo [6/7] Installing TensorFlow (Python 3.11)...
pip install tensorflow==2.15.0

echo [7/7] Installing MediaPipe (Python 3.11)...
pip install mediapipe==0.10.9

echo.
echo Creating project folders...
if not exist "cloud"       mkdir cloud
if not exist "detectors"   mkdir detectors
if not exist "video"       mkdir video
if not exist "models"      mkdir models
if not exist "saved_clips" mkdir saved_clips
if not exist "esp32"       mkdir esp32

type nul > cloud\__init__.py
type nul > detectors\__init__.py
type nul > video\__init__.py

echo.
echo Verifying installation...
echo.

python -c "import cv2;        print('OpenCV      OK:', cv2.__version__)"
python -c "import numpy;      print('NumPy       OK:', numpy.__version__)"
python -c "import fastapi;    print('FastAPI     OK')"
python -c "import supabase;   print('Supabase    OK')"
python -c "import mediapipe;  print('MediaPipe   OK')"
python -c "import tensorflow as tf; print('TensorFlow  OK:', tf.__version__)"
python -c "import dotenv;     print('python-dotenv OK')"

echo.
echo ================================================
echo    Setup Complete!
echo ================================================
echo.
echo Next steps:
echo 1. Check .env file has your credentials
echo 2. Copy models to models/ folder
echo 3. Run: python test_connection.py
echo 4. Run: python server.py
echo.
pause