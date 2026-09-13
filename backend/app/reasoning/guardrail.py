"""
SiteGuard AI - Part B Confidence & Visual Guardrails
Enforces strict quality gating, out-of-scope validation, and honest 'Insufficient Information' reporting.
"""

from typing import List, Dict, Any, Tuple
import cv2
import numpy as np


class ConfidenceGuardrail:
    BLUR_THRESHOLD = 55.0          # Laplacian variance < 55 indicates heavy blur
    LOW_LIGHT_THRESHOLD = 35.0     # Mean grayscale luminance < 35 indicates severe underexposure
    MIN_CONFIDENCE_GATE = 0.30     # Detections below 0.30 are considered uncertain

    @classmethod
    def evaluate(
        cls,
        image_bgr: np.ndarray,
        detections: List[Dict[str, Any]],
        question: str,
        intent: str
    ) -> Dict[str, Any]:
        """
        Evaluates visual input quality and detection certainty against query requirements.
        """
        if image_bgr is None or image_bgr.size == 0:
            return {
                "passed": False,
                "is_blurred": False,
                "is_low_light": False,
                "quality_score": 0.0,
                "confidence_gate_passed": False,
                "warning_notes": ["Image could not be decoded or is empty."],
                "insufficient_reason": "Image payload is missing or unreadable."
            }

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        mean_luminance = float(np.mean(gray))

        is_blurred = laplacian_var < cls.BLUR_THRESHOLD
        is_low_light = mean_luminance < cls.LOW_LIGHT_THRESHOLD

        # Calculate image quality index (0 - 100)
        blur_factor = min(1.0, laplacian_var / 120.0)
        light_factor = 1.0 - (abs(mean_luminance - 128.0) / 128.0) * 0.5
        quality_score = round(float(np.clip((blur_factor * 0.6 + light_factor * 0.4) * 100, 0, 100)), 1)

        warning_notes = []
        if is_blurred:
            warning_notes.append(f"Image is significantly blurred (Laplacian variance {laplacian_var:.1f} < {cls.BLUR_THRESHOLD})")
        if is_low_light:
            warning_notes.append(f"Image is severely underexposed (Mean luminance {mean_luminance:.1f} < {cls.LOW_LIGHT_THRESHOLD})")

        # Check Detection Confidence
        low_conf_detections = [d for d in detections if d.get("confidence", 0) < cls.MIN_CONFIDENCE_GATE]
        if low_conf_detections:
            warning_notes.append(f"{len(low_conf_detections)} detection(s) fell below minimum confidence gate ({cls.MIN_CONFIDENCE_GATE})")

        # Determine if information is insufficient for the specific question
        insufficient_reason = None

        if is_blurred and intent in ["PPE_COMPLIANCE", "VISUAL_COUNT"]:
            insufficient_reason = "Image motion blur prevents confident identification of fine PPE details (e.g. hardhat fitment or face masks)."

        elif is_low_light and intent in ["PPE_COMPLIANCE", "HAZARD_PROXIMITY"]:
            insufficient_reason = "Low-light conditions obscure worker apparel and equipment boundaries, preventing reliable safety assessment."

        elif intent == "PPE_COMPLIANCE" and not any(d["class_name"] == "Person" for d in detections) and detections:
            insufficient_reason = None

        elif len(detections) == 0 and intent in ["VISUAL_COUNT", "PPE_COMPLIANCE", "HAZARD_PROXIMITY"]:
            if is_blurred or is_low_light:
                insufficient_reason = "Due to poor image quality and zero confident detections, there is insufficient visual information to answer accurately."

        passed = (insufficient_reason is None) and (quality_score >= 25.0)

        return {
            "passed": passed,
            "is_blurred": is_blurred,
            "is_low_light": is_low_light,
            "quality_score": quality_score,
            "confidence_gate_passed": len(low_conf_detections) == 0,
            "warning_notes": warning_notes,
            "insufficient_reason": insufficient_reason
        }
