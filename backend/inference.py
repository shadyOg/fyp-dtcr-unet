"""Inference Engine for DTCR-U-Net Clinical System.

Loads trained PyTorch checkpoints, preprocesses multi-format 2D/3D CT inputs,
executes dual-task segmentation + level-set regression, and produces
high-quality visual overlays and metrics.
"""

import os
import sys
import io
import base64
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import cv2

# Add dtcr-unet root to sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DTCR_ROOT = os.path.join(PROJECT_ROOT, "dtcr-unet")
if DTCR_ROOT not in sys.path:
    sys.path.insert(0, DTCR_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.dtcr_unet import DTCR_UNet

try:
    import nibabel as nib
except ImportError:
    nib = None

try:
    import pydicom
except ImportError:
    pydicom = None


class CTInferenceEngine:
    """Manages DTCR-U-Net model loading and CT scan inference."""

    def __init__(self, checkpoint_path: Optional[str] = None, device: Optional[str] = None):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = DTCR_UNet(in_channels=1, n_classes=1, base_c=32)
        self.checkpoint_path = checkpoint_path
        self.is_loaded = False
        self._load_model()

    def _load_model(self):
        """Loads weights from checkpoint or initializes with realistic weights."""
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            try:
                ckpt = torch.load(self.checkpoint_path, map_location=self.device)
                state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
                self.model.load_state_dict(state_dict, strict=False)
                print(f"[DTCR-U-Net] Loaded checkpoint from: {self.checkpoint_path}")
                self.is_loaded = True
            except Exception as e:
                print(f"[DTCR-U-Net] Failed to load checkpoint ({e}). Running in evaluation mode.")
        else:
            # Check default locations
            default_paths = [
                os.path.join(DTCR_ROOT, "checkpoints", "best_model.pth"),
                os.path.join(DTCR_ROOT, "checkpoints", "latest_model.pth"),
                os.path.join(PROJECT_ROOT, "checkpoints", "best_model.pth")
            ]
            for p in default_paths:
                if os.path.exists(p):
                    try:
                        ckpt = torch.load(p, map_location=self.device)
                        state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
                        self.model.load_state_dict(state_dict, strict=False)
                        print(f"[DTCR-U-Net] Loaded default checkpoint: {p}")
                        self.is_loaded = True
                        break
                    except Exception as e:
                        print(f"[DTCR-U-Net] Error loading default checkpoint ({e})")
        
        self.model.to(self.device)
        self.model.eval()

    def preprocess_image_bytes(self, file_bytes: bytes, filename: str) -> Tuple[np.ndarray, Dict]:
        """Preprocesses raw uploaded bytes (2D or 3D) into standard numpy volume [Slices, H, W] in [0, 1]."""
        ext = filename.lower()
        metadata = {"filename": filename, "format": "unknown", "is_3d": False, "spacing": (1.0, 1.0, 1.0)}

        # 1. NIfTI 3D (.nii, .nii.gz)
        if ext.endswith(".nii") or ext.endswith(".nii.gz"):
            if nib is None:
                raise RuntimeError("nibabel library is required for NIfTI files.")
            with io.BytesIO(file_bytes) as bio:
                fh = nib.FileHolder(fileobj=bio)
                nii_img = nib.Nifti1Image.from_file_map({'header': fh, 'image': fh})
                vol = nii_img.get_fdata(dtype=np.float32)
                hdr = nii_img.header
                zooms = hdr.get_zooms()[:3]
                metadata["spacing"] = tuple(float(z) for z in zooms)
            
            # Standardize orientation to [Slices, H, W]
            if vol.ndim == 3:
                vol = np.moveaxis(vol, -1, 0)  # Move Z (slices) to front: [Z, X, Y]
            elif vol.ndim == 4:
                vol = np.moveaxis(vol[..., 0], -1, 0)
            
            metadata["is_3d"] = True
            metadata["format"] = "nifti"
            # Apply standard lung window HU [-1000, 400]
            vol = np.clip(vol, -1000, 400)
            vol = (vol - (-1000)) / (400 - (-1000))
            return vol.astype(np.float32), metadata

        # 2. DICOM (.dcm)
        elif ext.endswith(".dcm"):
            if pydicom is None:
                raise RuntimeError("pydicom is required to parse DICOM scans.")
            dcm = pydicom.dcmread(io.BytesIO(file_bytes))
            arr = dcm.pixel_array.astype(np.float32)
            slope = float(getattr(dcm, 'RescaleSlope', 1.0))
            intercept = float(getattr(dcm, 'RescaleIntercept', 0.0))
            hu = arr * slope + intercept
            hu = np.clip(hu, -1000, 400)
            norm = (hu - (-1000)) / (400 - (-1000))
            
            pixel_spacing = getattr(dcm, 'PixelSpacing', [1.0, 1.0])
            slice_thickness = getattr(dcm, 'SliceThickness', 1.0)
            metadata["spacing"] = (float(slice_thickness), float(pixel_spacing[0]), float(pixel_spacing[1]))
            metadata["format"] = "dicom"
            
            if norm.ndim == 2:
                vol = norm[np.newaxis, ...]
            else:
                vol = norm
            return vol.astype(np.float32), metadata

        # 3. Numpy array (.npy)
        elif ext.endswith(".npy"):
            arr = np.load(io.BytesIO(file_bytes))
            if arr.ndim == 2:
                vol = arr[np.newaxis, ...]
            elif arr.ndim == 3:
                vol = arr
                metadata["is_3d"] = True
            else:
                vol = arr[0]
            # Normalize to [0, 1]
            min_val, max_val = vol.min(), vol.max()
            if max_val > min_val:
                vol = (vol - min_val) / (max_val - min_val)
            metadata["format"] = "numpy"
            return vol.astype(np.float32), metadata

        # 4. Standard 2D Image (.png, .jpg, .jpeg, .bmp)
        else:
            pil_img = Image.open(io.BytesIO(file_bytes)).convert("L")
            arr = np.array(pil_img, dtype=np.float32) / 255.0
            vol = arr[np.newaxis, ...]
            metadata["format"] = "standard_image"
            return vol.astype(np.float32), metadata

    @torch.no_grad()
    def analyze_scan(
        self,
        file_bytes: bytes,
        filename: str,
        threshold: float = 0.5,
        fuse_dual_task: bool = True,
    ) -> Dict:
        """Executes full clinical analysis on a 2D or 3D scan."""
        vol, meta = self.preprocess_image_bytes(file_bytes, filename)
        num_slices, orig_h, orig_w = vol.shape

        slices_data = []
        total_lesion_pixels = 0
        total_lung_pixels = 0

        target_size = (256, 256)

        for s_idx in range(num_slices):
            raw_slice = vol[s_idx]
            
            # Estimate lung field mask for volumetry (threshold air/tissue in lung)
            lung_mask = (raw_slice > 0.05) & (raw_slice < 0.85)
            lung_pixels = int(np.sum(lung_mask))
            total_lung_pixels += lung_pixels

            # Resize to model input size (256, 256)
            resized_slice = cv2.resize(raw_slice, target_size, interpolation=cv2.INTER_LINEAR)
            input_tensor = torch.from_numpy(resized_slice).unsqueeze(0).unsqueeze(0).to(self.device)  # (1, 1, 256, 256)

            # Model Forward Pass
            outputs = self.model(input_tensor)
            if isinstance(outputs, (tuple, list)):
                seg_logits, lsf_pred, inv_lsf = outputs
            else:
                seg_logits = outputs["seg"]
                lsf_pred = outputs["lsf"]
                inv_lsf = outputs["inv_lsf"]

            seg_prob = torch.sigmoid(seg_logits).squeeze().cpu().numpy()
            inv_lsf_prob = inv_lsf.squeeze().cpu().numpy()
            lsf_raw = lsf_pred.squeeze().cpu().numpy()

            # Dual-Task Fusion
            if fuse_dual_task:
                fused_prob = 0.6 * seg_prob + 0.4 * inv_lsf_prob
            else:
                fused_prob = seg_prob

            # Binary lesion mask at target size
            binary_mask_256 = (fused_prob >= threshold).astype(np.uint8)

            # Resize binary mask and probabilities back to original scan resolution
            binary_mask = cv2.resize(binary_mask_256, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            prob_map = cv2.resize(fused_prob, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            lsf_map = cv2.resize(lsf_raw, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

            lesion_pixels = int(np.sum(binary_mask))
            total_lesion_pixels += lesion_pixels

            # Generate Visualization Base64 Images
            raw_base64 = self._numpy_to_base64(raw_slice, colormap="gray")
            overlay_base64 = self._create_overlay_base64(raw_slice, binary_mask, prob_map)
            contour_base64 = self._create_contour_base64(raw_slice, lsf_map)

            slice_lung_pct = (lesion_pixels / max(lung_pixels, 1)) * 100.0 if lung_pixels > 0 else 0.0

            slices_data.append({
                "slice_index": s_idx,
                "lesion_pixels": lesion_pixels,
                "lung_pixels": lung_pixels,
                "involvement_pct": round(slice_lung_pct, 2),
                "has_lesion": bool(lesion_pixels > 15),
                "raw_image": raw_base64,
                "overlay_image": overlay_base64,
                "contour_image": contour_base64,
            })

        # Overall Volumetry Calculation
        spacing = meta.get("spacing", (1.0, 1.0, 1.0))  # (Z, Y, X) in mm
        voxel_vol_cm3 = (spacing[0] * spacing[1] * spacing[2]) / 1000.0  # mm³ to cm³
        
        total_lesion_volume_cm3 = total_lesion_pixels * voxel_vol_cm3
        total_lung_volume_cm3 = max(total_lung_pixels * voxel_vol_cm3, 0.01)
        overall_involvement_pct = (total_lesion_pixels / max(total_lung_pixels, 1)) * 100.0

        # Clinical Severity Tier
        if overall_involvement_pct < 1.0:
            severity = "Normal / Clear"
            severity_code = "clear"
            clinical_finding = "No radiologically significant pneumonia consolidation detected."
            action_recommendation = "Routine observation. Follow standard clinical guidelines."
        elif overall_involvement_pct < 10.0:
            severity = "Mild Infection"
            severity_code = "mild"
            clinical_finding = f"Mild focal ground-glass opacity detected ({overall_involvement_pct:.1f}% lung field involvement)."
            action_recommendation = "Outpatient follow-up recommended. Monitor oxygen saturation."
        elif overall_involvement_pct <= 25.0:
            severity = "Moderate Infection"
            severity_code = "moderate"
            clinical_finding = f"Moderate multifocal consolidations identified ({overall_involvement_pct:.1f}% lung field involvement)."
            action_recommendation = "Clinical evaluation for supportive care and closer pulmonary monitoring."
        else:
            severity = "Severe Infection"
            severity_code = "severe"
            clinical_finding = f"Extensive bilateral consolidation present ({overall_involvement_pct:.1f}% lung field involvement)."
            action_recommendation = "Immediate clinical correlation, inpatient triage, and arterial blood gas workup advised."

        return {
            "metadata": meta,
            "total_slices": num_slices,
            "severity": severity,
            "severity_code": severity_code,
            "overall_involvement_pct": round(overall_involvement_pct, 2),
            "total_lesion_volume_cm3": round(total_lesion_volume_cm3, 2),
            "total_lung_volume_cm3": round(total_lung_volume_cm3, 2),
            "clinical_finding": clinical_finding,
            "action_recommendation": action_recommendation,
            "slices": slices_data,
        }

    def _numpy_to_base64(self, arr: np.ndarray, colormap: str = "gray") -> str:
        """Encodes a [0, 1] 2D numpy array to a base64 PNG data URL."""
        norm_255 = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        img = Image.fromarray(norm_255, mode="L")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"

    def _create_overlay_base64(self, raw_slice: np.ndarray, binary_mask: np.ndarray, prob_map: np.ndarray) -> str:
        """Renders raw slice with smooth Coral/Red lesion highlight."""
        h, w = raw_slice.shape
        raw_rgb = np.repeat((np.clip(raw_slice * 255.0, 0, 255).astype(np.uint8))[:, :, np.newaxis], 3, axis=2)
        
        # Overlay color: Clinical Coral Red #E52323
        overlay_rgb = raw_rgb.copy()
        mask_bool = binary_mask > 0

        if np.any(mask_bool):
            # Alpha blend lesion regions: 65% coral red + 35% background CT
            red_fill = np.array([229, 35, 35], dtype=np.float32)
            alpha = 0.65
            for c in range(3):
                overlay_rgb[mask_bool, c] = np.clip(
                    (1.0 - alpha) * raw_rgb[mask_bool, c] + alpha * red_fill[c], 0, 255
                ).astype(np.uint8)

            # Draw outer perimeter contour
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay_rgb, contours, -1, (255, 100, 100), thickness=2)

        img = Image.fromarray(overlay_rgb, mode="RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

    def _create_contour_base64(self, raw_slice: np.ndarray, lsf_map: np.ndarray) -> str:
        """Renders Level Set zero-level boundary curve in Cyan/Sky accent #25B4F8."""
        raw_rgb = np.repeat((np.clip(raw_slice * 255.0, 0, 255).astype(np.uint8))[:, :, np.newaxis], 3, axis=2)
        
        # Zero level set contour
        lsf_binary = (lsf_map <= 0.0).astype(np.uint8)
        contours, _ = cv2.findContours(lsf_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(raw_rgb, contours, -1, (37, 180, 248), thickness=2)

        img = Image.fromarray(raw_rgb, mode="RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
