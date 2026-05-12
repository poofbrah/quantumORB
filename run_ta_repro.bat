@echo off
setlocal

set "ROOT=%~dp0"
set "RAW=%ROOT%data\raw\Dataset_NQ_1min_2022_2025.csv"
set "PROCESSED=%ROOT%data\processed\nq_1min_2022_2025.parquet"

where python >nul 2>nul
if errorlevel 1 (
    echo.
    echo Python was not found on PATH.
    echo Activate the project environment first, then rerun this script.
    pause
    exit /b 1
)

echo.
echo === quantumORB TA Reproduction Runner ===
echo Repo root: %ROOT%

if exist "%PROCESSED%" (
    echo Found processed dataset:
    echo   %PROCESSED%
) else (
    if exist "%RAW%" (
        echo Processed dataset not found. Running preprocessing from raw CSV...
        python "%ROOT%examples\preprocess_nq_dataset.py"
        if errorlevel 1 (
            echo.
            echo Preprocessing failed.
            pause
            exit /b %errorlevel%
        )
    ) else (
        echo.
        echo Neither the processed parquet nor the raw CSV was found.
        echo Expected one of:
        echo   %PROCESSED%
        echo   %RAW%
        echo.
        echo Place Dataset_NQ_1min_2022_2025.csv in data\raw\ or add the processed parquet, then rerun.
        pause
        exit /b 1
    )
)

echo.
echo Running final paper experiment...
python "%ROOT%examples\run_final_paper_experiment.py"
if errorlevel 1 (
    echo.
    echo Final paper experiment failed.
    pause
    exit /b %errorlevel%
)

echo.
echo Optional verification: running test suite...
python -m pytest -q
if errorlevel 1 (
    echo.
    echo Tests failed.
    pause
    exit /b %errorlevel%
)

echo.
echo Reproduction completed successfully.
echo Review these files:
echo   %ROOT%outputs\final_paper_experiment\paper_experiment_comparison.csv
echo   %ROOT%outputs\final_paper_experiment\paper_experiment_summary.json
echo   %ROOT%outputs\final_paper_experiment\figures\
pause
