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


def make_synthetic_thoracic_ct():
    """Generates a synthetic axial chest CT slice with chest wall and bilateral lungs."""
    y, x = np.ogrid[:256, :256]
    body = (x - 128) ** 2 / (105 ** 2) + (y - 128) ** 2 / (85 ** 2) <= 1.0
    ct = np.zeros((256, 256), dtype=np.uint8)
    ct[body] = 160  # soft tissue chest wall

    # Bilateral lungs (left and right low-density air cavities)
    left_lung = (x - 85) ** 2 / (35 ** 2) + (y - 128) ** 2 / (45 ** 2) <= 1.0
    right_lung = (x - 171) ** 2 / (35 ** 2) + (y - 128) ** 2 / (45 ** 2) <= 1.0
    ct[left_lung] = 35
    ct[right_lung] = 35

    # Simulated consolidation lesion inside right lung
    lesion = (x - 165) ** 2 / (16 ** 2) + (y - 120) ** 2 / (16 ** 2) <= 1.0
    ct[lesion] = 190
    return ct


def make_synthetic_head_ct():
    """Generates a synthetic axial head/brain CT slice (circular cranium with brain tissue, no lungs)."""
    y, x = np.ogrid[:256, :256]
    skull = (x - 128) ** 2 + (y - 128) ** 2 <= (90 ** 2)
    brain = (x - 128) ** 2 + (y - 128) ** 2 <= (80 ** 2)
    ct = np.zeros((256, 256), dtype=np.uint8)
    ct[skull] = 240  # Dense bone / skull
    ct[brain] = 130  # Brain parenchyma (solid, no lung cavities)
    return ct


def test_instant_preview_and_validation():
    print("[Test 1] Testing Instant Scan Preview & Validation on Valid Thoracic CT...")
    engine = CTInferenceEngine()

    synthetic_ct = make_synthetic_thoracic_ct()
    buf = io.BytesIO()
    np.save(buf, synthetic_ct)
    npy_bytes = buf.getvalue()

    preview_res = engine.preview_scan(npy_bytes, filename="thoracic_scan.npy")
    assert preview_res["is_preview"] is True
    assert preview_res["is_valid_ct"] is True
    assert preview_res["validation_status"] == "valid"
    assert "raw_image" in preview_res["slices"][0]
    print(f"  [PASSED] Valid Thoracic CT accepted (Reason: {preview_res['validation_reason']})")


def test_non_lung_ct_accepted():
    print("\n[Test 1b] Testing Non-Lung CT Acceptance (Head / Brain / Abdomen CT)...")
    engine = CTInferenceEngine()

    head_ct = make_synthetic_head_ct()
    buf = io.BytesIO()
    Image.fromarray(head_ct).save(buf, format="PNG")
    png_bytes = buf.getvalue()

    preview_res = engine.preview_scan(png_bytes, filename="head_brain_ct.png")
    assert preview_res["is_valid_ct"] is True
    assert preview_res["validation_status"] == "valid"
    print(f"  [PASSED] Non-Lung CT scan accepted! (Reason: {preview_res['validation_reason']})")


def test_color_photo_rejection():
    print("\n[Test 2] Testing Guardrail: Rejecting Non-Medical Color Photograph (Selfie / Table)...")
    engine = CTInferenceEngine()

    # Create an RGB color image with distinct red, green, blue color channels
    rgb_img = np.zeros((256, 256, 3), dtype=np.uint8)
    rgb_img[:, :, 0] = 220  # red tint
    rgb_img[:, :, 1] = 120  # green tint
    rgb_img[:, :, 2] = 50   # blue tint

    buf = io.BytesIO()
    Image.fromarray(rgb_img).save(buf, format="JPEG")
    jpg_bytes = buf.getvalue()

    # 1. Preview mode should flag invalid scan
    preview_res = engine.preview_scan(jpg_bytes, filename="selfie_table_photo.jpg")
    assert preview_res["is_valid_ct"] is False
    assert preview_res["validation_status"] == "invalid_scan"
    assert preview_res["validation_reason"] == "Invalid image type, please upload a valid CT scan."
    print(f"  [PASSED] Color Photo Guardrail triggered in Preview: {preview_res['validation_reason']}")

    # 2. Analyze mode should raise ValueError
    try:
        engine.analyze_scan(jpg_bytes, filename="selfie_table_photo.jpg")
        assert False, "Expected ValueError on color photograph"
    except ValueError as ve:
        assert str(ve) == "Invalid image type, please upload a valid CT scan."
        print(f"  [PASSED] Analysis safely rejected color photo: {ve}")


def test_grayscale_non_ct_rejection():
    print("\n[Test 3] Testing Guardrail: Rejecting Grayscale Non-Medical Image (Wall / Texture)...")
    engine = CTInferenceEngine()

    # Uniform or textured image filling the entire field of view (e.g., photo of a wall/table converted to grayscale)
    non_ct = np.full((256, 256), 180, dtype=np.uint8)

    buf = io.BytesIO()
    Image.fromarray(non_ct).save(buf, format="PNG")
    png_bytes = buf.getvalue()

    preview_res = engine.preview_scan(png_bytes, filename="wall_texture.png")
    assert preview_res["is_valid_ct"] is False
    assert preview_res["validation_status"] == "invalid_scan"
    assert preview_res["validation_reason"] == "Invalid image type, please upload a valid CT scan."
    print(f"  [PASSED] Non-CT Grayscale Guardrail triggered: {preview_res['validation_reason']}")


def test_2d_analysis_and_ground_truth():
    print("\n[Test 4] Testing 2D CT Slice Inference & Ground Truth Comparison on Valid Scan...")
    engine = CTInferenceEngine()

    synthetic_ct = make_synthetic_thoracic_ct()
    buf = io.BytesIO()
    Image.fromarray(synthetic_ct).save(buf, format="PNG")
    png_bytes = buf.getvalue()

    result = engine.analyze_scan(png_bytes, filename="valid_chest_scan.png")
    assert result["total_slices"] == 1
    assert result["is_valid_ct"] is True
    assert "severity" in result
    print("  [PASSED] 2D Analysis successfully completed on valid scan!")

    # Test Ground Truth Comparison
    y, x = np.ogrid[:256, :256]
    gt_mask = np.zeros((256, 256), dtype=np.uint8)
    gt_mask[(x - 165) ** 2 / (16 ** 2) + (y - 120) ** 2 / (16 ** 2) <= 1.0] = 1

    mask_buf = io.BytesIO()
    np.save(mask_buf, gt_mask)
    gt_npy_bytes = mask_buf.getvalue()

    comparison_result = engine.compare_ground_truth(
        mask_bytes=gt_npy_bytes,
        filename="test_ground_truth.npy",
        analysis_data=result
    )
    assert comparison_result["has_ground_truth"] is True
    assert "overall_dice" in comparison_result
    assert "ai_accuracy" in comparison_result
    assert "accuracy_tier" in comparison_result
    assert "accuracy_interpretation" in comparison_result
    assert "slice_ai_accuracy" in comparison_result["slices"][0]
    print(f"  -> Model Accuracy: {comparison_result['ai_accuracy']}% ({comparison_result['accuracy_tier']})")
    print("  [PASSED] Ground Truth comparison completed!")

    return comparison_result


def test_pdf_report_generation(analysis_result):
    print("\n[Test 5] Testing Medical PDF Diagnostic Report Generation...")
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

    print(f"  -> Generated PDF size: {len(pdf_bytes)} bytes")
    print("  [PASSED] Medical PDF Report successfully generated!")


if __name__ == "__main__":
    test_instant_preview_and_validation()
    test_non_lung_ct_accepted()
    test_color_photo_rejection()
    test_grayscale_non_ct_rejection()
    res = test_2d_analysis_and_ground_truth()
    test_pdf_report_generation(res)
    print("\nAll End-to-End Clinical Interface & Guardrail Tests Passed Successfully! [OK]")
