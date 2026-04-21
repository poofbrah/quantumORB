@echo off
setlocal
set ROOT=%~dp0
python "%ROOT%examples\run_final_paper_experiment.py" %*
if errorlevel 1 (
    echo.
    echo Final paper experiment failed.
    pause
    exit /b %errorlevel%
)
echo.
echo Final paper experiment completed successfully.
pause
