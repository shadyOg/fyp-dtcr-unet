@echo off
title DTCR-U-Net Clinical Platform Launcher
echo ========================================================
echo   DTCR-U-Net: Clinical CT Lesion Segmentation Platform
echo ========================================================
echo.

cd /d "%~dp0"

echo [1/3] Checking Python Virtual Environment...
if not exist ".venv\Scripts\python.exe" (
    echo Creating .venv virtual environment...
    python -m venv .venv
    call .venv\Scripts\activate.bat
    pip install -r backend\requirements.txt
) else (
    echo .venv found.
)

echo [2/3] Launching FastAPI Backend on http://localhost:8000 ...
start "DTCR-U-Net Backend (Port 8000)" cmd /k "call .venv\Scripts\activate.bat && cd backend && python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload"

echo [3/3] Launching React Frontend on http://localhost:3000 ...
start "DTCR-U-Net Frontend (Port 3000)" cmd /k "cd frontend && npm.cmd run dev"

echo.
echo ========================================================
echo   Workstation is launching!
echo   Frontend: http://localhost:3000
echo   Backend API: http://localhost:8000/docs
echo ========================================================
echo.
timeout /t 3 >nul
start http://localhost:3000
