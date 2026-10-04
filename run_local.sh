#!/usr/bin/env bash
set -e

echo "========================================================"
echo "  DTCR-U-Net: Clinical CT Lesion Segmentation Platform"
echo "========================================================"
echo ""

# Ensure current directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

# 1. Virtual Environment
if [ ! -d ".venv" ]; then
    echo "[1/3] Creating Python virtual environment .venv..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r backend/requirements.txt
else
    echo "[1/3] Virtual environment .venv found."
    source .venv/bin/activate
fi

# 2. Start Backend
echo "[2/3] Starting FastAPI Backend on port 8000..."
(cd backend && python -m uvicorn main:app --host 0.0.0.0 --port 8000) &
BACKEND_PID=$!

# 3. Start Frontend
echo "[3/3] Starting React Frontend on port 3000..."
(cd frontend && npm run dev) &
FRONTEND_PID=$!

echo ""
echo "========================================================"
echo "  Workstation running!"
echo "  Frontend: http://localhost:3000"
echo "  Backend API: http://localhost:8000/docs"
echo "========================================================"
echo "Press Ctrl+C to terminate both servers."

trap "kill $BACKEND_PID $FRONTEND_PID" EXIT
wait
