@echo off
where python >nul 2>nul
IF %ERRORLEVEL% NEQ 0 (
  echo Python not found in PATH. Install Python 3.10+ and re-run.
  exit /b 1
)
python -m venv demand_forecasting_env
call demand_forecasting_env\Scripts\activate
echo Venv activated. Now run: pip install -r requirements.txt
