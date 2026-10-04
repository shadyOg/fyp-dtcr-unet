"""End-to-End Test for DTCR-U-Net Clinical Web API and Inference Engine."""

import os
import sys
import io
import numpy as np
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "backend"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "dtcr-unet"))

from inference import CTInferenceEngine
from report_generator import ClinicalDiagnosticReport


def test_2d_analysis():
    print("[Test 1] Testing 2D CT Slice Inference & Overlay Generation...")
    engine = CTInferenceEngine()

    # Create synthetic 2D CT slice (256x256) with simulated lung circle and lesion
    synthetic_ct = np.zeros((256, 256), dtype=np.uint8)
    # Lung field
    y, x = np.ogrid[:256, :256]
    lung_mask = (x - 128) ** 2 + (y - 128) ** 2 <= 90 ** 2
    synthetic_ct[lung_mask] = 120
    # Simulated lesion
    lesion_mask = (x - 150) ** 2 + (y - 140) ** 2 <= 25 ** 2
    synthetic_ct[lesion_mask] = 210

    # Save to PNG bytes
    buf = io.BytesIO()
    Image.fromarray(synthetic_ct).save(buf, format="PNG")
    png_bytes = buf.getvalue()

    result = engine.analyze_scan(png_bytes, filename="test_scan.png")

    assert result["total_slices"] == 1, "Expected 1 slice"
    assert "severity" in result, "Missing severity field"
    assert "overall_involvement_pct" in result, "Missing involvement percentage"
    assert "slices" in result and len(result["slices"]) == 1, "Missing slice outputs"
    
    slice_0 = result["slices"][0]
    assert slice_0["raw_image"].startswith("data:image/png;base64,"), "Invalid raw image data URL"
    assert slice_0["overlay_image"].startswith("data:image/png;base64,"), "Invalid overlay data URL"
    assert slice_0["contour_image"].startswith("data:image/png;base64,"), "Invalid contour data URL"

    print(f"  -> Severity: {result['severity']} ({result['overall_involvement_pct']}% involvement)")
    print(f"  -> Lesion Volume: {result['total_lesion_volume_cm3']} cm3")
    print("  [PASSED] 2D Analysis successfully completed!")

    return result


def test_pdf_report_generation(analysis_result):
    print("\n[Test 2] Testing Medical PDF Diagnostic Report Generation...")
    reporter = ClinicalDiagnosticReport(
        analysis_result,
        patient_info={
            "patient_id": "TEST-PT-101",
            "patient_name": "Test Patient",
            "age": "62",
            "gender": "Female",
            "referring_physician": "Dr. Smith",
        }
    )

    pdf_bytes = reporter.generate_pdf_bytes()
    assert len(pdf_bytes) > 1000, "PDF bytes should be non-empty"
    assert pdf_bytes.startswith(b"%PDF"), "Output is not a valid PDF document"

    test_pdf_path = os.path.join(PROJECT_ROOT, "test_output_report.pdf")
    with open(test_pdf_path, "wb") as f:
        f.write(pdf_bytes)

    print(f"  -> Generated PDF size: {len(pdf_bytes)} bytes")
    print(f"  -> Saved test sample to: {test_pdf_path}")
    print("  [PASSED] Medical PDF Report successfully generated!")


if __name__ == "__main__":
    res = test_2d_analysis()
    test_pdf_report_generation(res)
    print("\nAll End-to-End Clinical Interface Tests Passed Successfully! [OK]")
