@echo off
setlocal
set ROOT=%~dp0
python "%ROOT%examples\run_setup_quality_research_pipeline.py" %*
if errorlevel 1 (
    echo.
    echo Pipeline failed.
    pause
    exit /b %errorlevel%
)
echo.
echo Pipeline completed successfully.
pause
