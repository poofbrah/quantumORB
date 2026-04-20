@echo off
setlocal
set ROOT=%~dp0
"C:\Users\prana\AppData\Local\Programs\Python\Python311\python.exe" "%ROOT%examples\run_setup_quality_research_pipeline.py"
if errorlevel 1 (
    echo.
    echo Pipeline failed.
    pause
    exit /b %errorlevel%
)
echo.
echo Pipeline completed successfully.
pause
