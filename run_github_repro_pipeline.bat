@echo off
setlocal
set ROOT=%~dp0
python "%ROOT%examples\run_github_repro_pipeline.py" %*
if errorlevel 1 (
    echo.
    echo Repro pipeline failed.
    pause
    exit /b %errorlevel%
)
echo.
echo Repro pipeline completed successfully.
pause
