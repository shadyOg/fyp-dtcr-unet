"""FastAPI Server for DTCR-U-Net Clinical Segmentation Platform.

Provides REST endpoints for:
- Health check & model status
- POST /api/analyze: Multi-format CT image/volume upload and dual-task inference
- POST /api/export-report: Generates downloadable clinical PDF diagnostic report
"""

import os
import sys
from typing import Optional, Dict
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
import uvicorn

# Local module imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from inference import CTInferenceEngine
from report_generator import ClinicalDiagnosticReport

app = FastAPI(
    title="DTCR-U-Net Clinical API",
    description="Dual-Task Consistency Regularized U-Net for Pneumonia Lesion Segmentation",
    version="1.0.0"
)

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Inference Engine
engine = CTInferenceEngine()


@app.get("/api/status")
def status():
    return {
        "status": "online",
        "service": "DTCR-U-Net Clinical Segmentation API",
        "device": str(engine.device),
        "model_loaded": engine.is_loaded
    }


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "device": str(engine.device),
        "model_loaded": engine.is_loaded
    }


@app.post("/api/preview")
async def preview_scan(
    file: UploadFile = File(...)
):
    """Receives uploaded CT file (.npy, .dcm, .nii, .png, etc.) and parses slices for immediate inspection without running inference."""
    try:
        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        filename = file.filename or "scan.npy"
        results = engine.preview_scan(
            file_bytes=file_bytes,
            filename=filename
        )
        return JSONResponse(content={"success": True, "data": results})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": f"Failed to preview scan: {str(e)}"}
        )


@app.post("/api/analyze")
async def analyze_scan(
    file: UploadFile = File(...),
    threshold: float = Form(0.5),
    fuse_dual_task: bool = Form(True)
):
    """Receives uploaded CT file (2D image, DICOM, or 3D NIfTI) and executes model inference."""
    try:
        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        filename = file.filename or "scan.png"
        results = engine.analyze_scan(
            file_bytes=file_bytes,
            filename=filename,
            threshold=threshold,
            fuse_dual_task=fuse_dual_task
        )
        return JSONResponse(content={"success": True, "data": results})

    except ValueError as ve:
        return JSONResponse(
            status_code=422,
            content={"success": False, "error": str(ve)}
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )


@app.post("/api/compare-ground-truth")
async def compare_ground_truth(
    mask_file: UploadFile = File(...),
    analysis_json: str = Form(...)
):
    """Receives uploaded Ground Truth mask (e.g. .npy, .png, .dcm, .nii) and computes visual comparison & Dice metrics."""
    try:
        import json
        mask_bytes = await mask_file.read()
        if not mask_bytes:
            raise HTTPException(status_code=400, detail="Uploaded mask file is empty.")

        analysis_data = json.loads(analysis_json)
        filename = mask_file.filename or "mask.npy"

        comparison_results = engine.compare_ground_truth(
            mask_bytes=mask_bytes,
            filename=filename,
            analysis_data=analysis_data
        )
        return JSONResponse(content={"success": True, "data": comparison_results})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": f"Failed to compare mask: {str(e)}"}
        )


@app.post("/api/export-report")
async def export_report(payload: Dict):
    """Generates a downloadable clinical PDF diagnostic report from analysis findings."""
    try:
        analysis_data = payload.get("analysis_data", {})
        patient_info = payload.get("patient_info", None)

        if not analysis_data:
            raise HTTPException(status_code=400, detail="Missing analysis data for report generation.")

        reporter = ClinicalDiagnosticReport(analysis_data, patient_info)
        pdf_bytes = reporter.generate_pdf_bytes()

        patient_id = (patient_info or {}).get("patient_id", "Patient")
        filename = f"DTCR_Report_{patient_id}.pdf"

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {str(e)}")


# ---------------------------------------------------------------------------
# Serve React Frontend SPA (Unified Service)
# ---------------------------------------------------------------------------
FRONTEND_DIST = os.path.join(PROJECT_ROOT, "frontend", "dist")
if not os.path.isdir(FRONTEND_DIST):
    FRONTEND_DIST = os.path.join(os.getcwd(), "frontend", "dist")

if os.path.isdir(FRONTEND_DIST):
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse

    assets_dir = os.path.join(FRONTEND_DIST, "assets")
    if os.path.isdir(assets_dir):
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/")
    async def serve_index():
        return FileResponse(os.path.join(FRONTEND_DIST, "index.html"))

    @app.get("/index.html")
    async def serve_index_html():
        return FileResponse(os.path.join(FRONTEND_DIST, "index.html"))
else:
    @app.get("/")
    def fallback_root():
        return {
            "status": "online",
            "service": "DTCR-U-Net Clinical Segmentation API (Frontend build not found)",
            "device": str(engine.device),
            "model_loaded": engine.is_loaded
        }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"[DTCR-U-Net API] Starting server on http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
