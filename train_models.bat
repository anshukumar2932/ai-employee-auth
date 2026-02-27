@echo off
REM Gait Identification Model Training Script (Windows)
REM This script trains all models and generates comparison reports

echo ==========================================
echo Gait Identification Model Training
echo ==========================================
echo.

REM Check if data exists
if not exist "data\cleaned_walking_data" (
    echo ERROR: Cleaned data not found!
    echo Please run the data preparation notebook first:
    echo   jupyter notebook notebooks\gait_pipeline.ipynb
    exit /b 1
)

REM Create necessary directories
if not exist "results" mkdir results
if not exist "models" mkdir models

echo [OK] Data found
echo [OK] Directories created
echo.

REM Check Python dependencies
echo Checking dependencies...
python -c "import sklearn, torch, numpy, pandas" 2>nul
if errorlevel 1 (
    echo Missing dependencies. Installing...
    pip install -r requirements.txt
) else (
    echo [OK] All dependencies installed
)

echo.
echo ==========================================
echo Starting Model Training
echo ==========================================
echo.
echo This will train 4 models:
echo   1. Logistic Regression (baseline)
echo   2. Random Forest (best simple model)
echo   3. SVM (best for small data)
echo   4. Simple 1D CNN (deep learning)
echo.
echo Estimated time: 10-30 minutes
echo.

REM Run training
python src\train_gait_models.py

REM Check if training was successful
if errorlevel 1 (
    echo.
    echo Training failed. Check the error messages above.
    exit /b 1
)

echo.
echo ==========================================
echo Training Complete!
echo ==========================================
echo.
echo Results saved to:
echo   results\model_comparison.csv
echo   results\model_comparison.png
echo   results\*_confusion_matrix.png
echo   models\best_model_metadata.json
echo.
echo To view results:
echo   type results\model_comparison.csv
echo   start results\model_comparison.png
echo.

pause
