"""Medical PDF Diagnostic Report Generator for DTCR-U-Net.

Creates an elegant, clinician-ready radiology report containing:
- Patient metadata & exam header with custom brand header (#7000A5 / #011632)
- Clinical volumetry summary & severity score badge
- High-contrast visual thumbnails of key lesion slices
- Detailed plain-English diagnostic remarks
"""

import io
import os
import base64
from datetime import datetime
from typing import Dict, List, Optional
from PIL import Image

try:
    from fpdf import FPDF
except ImportError:
    FPDF = None

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


class ClinicalDiagnosticReport:
    """Builds a formatted PDF report from analysis results."""

    def __init__(self, analysis_result: Dict, patient_info: Optional[Dict] = None):
        self.data = analysis_result
        self.patient = patient_info or {
            "patient_id": "PT-" + datetime.now().strftime("%Y%m%d-%H%M"),
            "patient_name": "Anonymous / De-identified",
            "age": "N/A",
            "gender": "N/A",
            "referring_physician": "Attending Pulmonologist",
        }

    def generate_pdf_bytes(self) -> bytes:
        """Renders the clinical PDF report and returns raw bytes."""
        if FPDF is None:
            raise RuntimeError("fpdf2 is required to generate PDF reports.")

        pdf = FPDF(orientation="P", unit="mm", format="A4")
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        # Header Banner (#7000A5 & #011632)
        pdf.set_fill_color(112, 0, 165)  # #7000A5
        pdf.rect(0, 0, 210, 28, "F")
        
        pdf.set_font("Helvetica", "B", 18)
        pdf.set_text_color(255, 255, 255)
        pdf.set_xy(12, 6)
        pdf.cell(0, 8, "DTCR-U-Net Clinical Diagnostic Report", ln=True)
        
        pdf.set_font("Helvetica", "", 10)
        pdf.set_xy(12, 16)
        pdf.cell(0, 6, "AI-Assisted Pulmonary CT Lesion Segmentation & Volumetry", ln=True)

        pdf.set_xy(140, 16)
        pdf.cell(0, 6, f"Exam Date: {datetime.now().strftime('%b %d, %Y - %H:%M')}", ln=True, align="R")

        pdf.ln(12)

        # Patient Information Table Card
        pdf.set_fill_color(248, 249, 250)
        pdf.set_draw_color(220, 224, 230)
        pdf.rect(12, 34, 186, 26, "DF")

        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(1, 22, 50)  # #011632
        pdf.set_xy(16, 38)
        pdf.cell(40, 6, f"Patient ID: {self.patient.get('patient_id')}")
        pdf.cell(70, 6, f"Name: {self.patient.get('patient_name')}")
        pdf.cell(40, 6, f"Gender/Age: {self.patient.get('gender')}/{self.patient.get('age')}", ln=True)

        pdf.set_xy(16, 46)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(84, 87, 90)
        pdf.cell(70, 6, f"Scan File: {self.data.get('metadata', {}).get('filename', 'scan.dcm')}")
        pdf.cell(60, 6, f"Total Slices: {self.data.get('total_slices', 1)}")
        pdf.cell(40, 6, f"Ref Physician: {self.patient.get('referring_physician')}", ln=True)

        pdf.ln(12)

        # Executive Clinical Summary Box
        severity_code = self.data.get("severity_code", "clear")
        if severity_code == "clear":
            box_r, box_g, box_b = 23, 191, 40    # Green #17BF28
            tag_bg = (235, 252, 237)
        elif severity_code == "mild":
            box_r, box_g, box_b = 37, 180, 248   # Blue #25B4F8
            tag_bg = (230, 246, 254)
        elif severity_code == "moderate":
            box_r, box_g, box_b = 236, 148, 44   # Amber #EC942C
            tag_bg = (254, 247, 237)
        else:
            box_r, box_g, box_b = 229, 35, 35    # Red #E52323
            tag_bg = (254, 238, 238)

        pdf.set_fill_color(*tag_bg)
        pdf.set_draw_color(box_r, box_g, box_b)
        pdf.rect(12, 66, 186, 32, "DF")

        pdf.set_xy(16, 70)
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(box_r, box_g, box_b)
        pdf.cell(100, 6, f"Diagnostic Assessment: {self.data.get('severity', 'Unknown')}")
        
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(1, 22, 50)
        pdf.cell(70, 6, f"Lung Involvement: {self.data.get('overall_involvement_pct', 0.0)}%", ln=True, align="R")

        pdf.set_xy(16, 78)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(51, 51, 51)
        pdf.multi_cell(178, 5, f"{self.data.get('clinical_finding', '')}")

        pdf.ln(10)

        # Quantitative Volumetry Metrics Grid
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(1, 22, 50)
        pdf.set_xy(12, 104)
        pdf.cell(0, 6, "Quantitative Volumetry Findings", ln=True)

        pdf.set_draw_color(220, 224, 230)
        pdf.set_fill_color(255, 255, 255)
        
        # Metric 1
        pdf.rect(12, 112, 58, 20, "DF")
        pdf.set_xy(14, 114)
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(120, 120, 120)
        pdf.cell(54, 4, "TOTAL LESION VOLUME", ln=True)
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(112, 0, 165)
        pdf.set_xy(14, 120)
        pdf.cell(54, 6, f"{self.data.get('total_lesion_volume_cm3', 0.0)} cm³", ln=True)

        # Metric 2
        pdf.rect(76, 112, 58, 20, "DF")
        pdf.set_xy(78, 114)
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(120, 120, 120)
        pdf.cell(54, 4, "TOTAL LUNG FIELD", ln=True)
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(19, 118, 248)
        pdf.set_xy(78, 120)
        pdf.cell(54, 6, f"{self.data.get('total_lung_volume_cm3', 0.0)} cm³", ln=True)

        # Metric 3
        pdf.rect(140, 112, 58, 20, "DF")
        pdf.set_xy(142, 114)
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(120, 120, 120)
        pdf.cell(54, 4, "INVOLVEMENT RATIO", ln=True)
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(box_r, box_g, box_b)
        pdf.set_xy(142, 120)
        pdf.cell(54, 6, f"{self.data.get('overall_involvement_pct', 0.0)} %", ln=True)

        pdf.ln(18)

        # Key Representative Slices Visual Section
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(1, 22, 50)
        pdf.set_xy(12, 138)
        pdf.cell(0, 6, "Representative CT Slices & Highlighted Lesion Detections", ln=True)

        slices = self.data.get("slices", [])
        # Pick up to 3 informative slices (prefer slices with highest lesion involvement)
        sorted_slices = sorted(slices, key=lambda s: s.get("involvement_pct", 0), reverse=True)
        sample_slices = sorted_slices[:3] if len(sorted_slices) >= 3 else slices[:3]

        img_y = 146
        col_width = 58
        for i, s in enumerate(sample_slices):
            col_x = 12 + i * (col_width + 6)
            overlay_b64 = s.get("overlay_image", "")
            if overlay_b64 and "base64," in overlay_b64:
                raw_bytes = base64.b64decode(overlay_b64.split("base64,")[1])
                pil_img = Image.open(io.BytesIO(raw_bytes))
                
                # Save temp image for FPDF
                temp_path = os.path.join(PROJECT_ROOT, f"temp_slice_{i}.png")
                pil_img.save(temp_path)
                try:
                    pdf.image(temp_path, x=col_x, y=img_y, w=col_width, h=col_width)
                    pdf.set_xy(col_x, img_y + col_width + 2)
                    pdf.set_font("Helvetica", "B", 8)
                    pdf.set_text_color(51, 51, 51)
                    pdf.cell(col_width, 4, f"Slice #{s.get('slice_index', i) + 1} ({s.get('involvement_pct', 0)}%)", align="C")
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

        pdf.ln(col_width + 12)

        # Clinical Recommendation Footer Box
        pdf.set_xy(12, 224)
        pdf.set_fill_color(245, 247, 250)
        pdf.set_draw_color(210, 215, 225)
        pdf.rect(12, 224, 186, 26, "DF")

        pdf.set_xy(16, 228)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(112, 0, 165)
        pdf.cell(0, 4, "Clinical Recommendation & Next Steps:", ln=True)

        pdf.set_xy(16, 234)
        pdf.set_font("Helvetica", "", 8.5)
        pdf.set_text_color(60, 73, 89)
        pdf.multi_cell(178, 4, f"{self.data.get('action_recommendation', 'Clinical follow-up recommended.')}")

        # Disclaimer
        pdf.set_xy(12, 258)
        pdf.set_font("Helvetica", "I", 7.5)
        pdf.set_text_color(151, 151, 151)
        pdf.multi_cell(
            186, 3.5,
            "Notice: This report is generated by DTCR-U-Net AI as a diagnostic aid. It must be correlated with patient history, physical examination, and professional radiological judgment before clinical decisions."
        )

        return bytes(pdf.output())
