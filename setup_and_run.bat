@echo off
REM Startup script for Plant Disease Detection (Windows)
REM Usage: setup_and_run.bat

setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ========================================
echo Plant Disease Detection - Setup ^& Run
echo ========================================

REM Check Python
echo.
echo Checking Python version...
python --version
if errorlevel 1 (
    echo Error: Python not found. Install Python 3.9+ from https://www.python.org/
    pause
    exit /b 1
)

REM Create virtual environment
echo.
echo Creating virtual environment...
if not exist venv (
    python -m venv venv
    echo [OK] Virtual environment created
) else (
    echo [OK] Virtual environment already exists
)

REM Activate virtual environment
echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo [OK] Activated

REM Install dependencies
echo.
echo Installing dependencies...
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
echo [OK] Dependencies installed

REM Create directories
echo.
echo Creating project directories...
if not exist datasets\raw mkdir datasets\raw
if not exist datasets\yolo_format mkdir datasets\yolo_format
if not exist models mkdir models
if not exist backend\uploads mkdir backend\uploads
if not exist frontend\templates mkdir frontend\templates
if not exist frontend\static mkdir frontend\static
if not exist notebooks mkdir notebooks
echo [OK] Directories created

REM Check dataset
echo.
echo Checking dataset...
if not exist "datasets\raw\*" (
    echo [WARNING] PlantVillage dataset not found
    echo.
    echo Download from: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
    echo Extract to: datasets\raw\
    echo.
    echo Then run: python utils/dataset_converter.py
) else (
    echo [OK] Dataset found
    echo.
    echo Converting dataset to YOLO format...
    python utils/dataset_converter.py
)

REM Check model
echo.
echo Checking model...
if not exist "models\best.pt" (
    echo [WARNING] Model not found
    echo.
    echo Options:
    echo 1. Download pre-trained: 
    echo    wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt -O models/best.pt
    echo 2. Train your own: 
    echo    python utils/train_yolov5.py
) else (
    echo [OK] Model found
)

REM Start Flask server
echo.
echo ========================================
echo Starting Flask server...
echo ========================================
echo.
echo Server will be available at: http://localhost:5000
echo Press Ctrl+C to stop
echo.

cd backend
python app.py

pause
