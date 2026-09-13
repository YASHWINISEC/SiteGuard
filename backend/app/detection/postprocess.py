"""
SiteGuard AI - Detection Postprocessing Module
Handles bounding box scaling, NMS filtering, visual annotation, and distance calculations.
"""

from typing import List, Dict, Any, Tuple
import cv2
import numpy as np
import math

# 25 Construction Site Safety Classes
CLASSES = [
    'Excavator', 'Gloves', 'Hardhat', 'Ladder', 'Mask',
    'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'SUV',
    'Safety Cone', 'Safety Vest', 'bus', 'dump truck', 'fire hydrant',
    'machinery', 'mini-van', 'sedan', 'semi', 'trailer',
    'truck', 'truck and trailer', 'van', 'vehicle', 'wheel loader'
]

PPE_VIOLATIONS = {"NO-Hardhat", "NO-Safety Vest", "NO-Mask"}
HEAVY_EQUIPMENT = {"Excavator", "dump truck", "wheel loader", "machinery", "truck and trailer", "semi", "trailer", "truck"}

CLASS_COLORS = {
    "Person": "#3B82F6",
    "Hardhat": "#10B981",
    "Safety Vest": "#10B981",
    "Mask": "#06B6D4",
    "Gloves": "#8B5CF6",
    "NO-Hardhat": "#EF4444",
    "NO-Safety Vest": "#EF4444",
    "NO-Mask": "#F59E0B",
    "Excavator": "#F97316",
    "dump truck": "#F97316",
    "wheel loader": "#F97316",
    "machinery": "#EA580C",
    "truck": "#D97706",
    "truck and trailer": "#D97706",
    "semi": "#D97706",
    "trailer": "#D97706",
    "Ladder": "#6366F1",
    "Safety Cone": "#EAB308",
    "fire hydrant": "#EC4899",
    "sedan": "#64748B",
    "SUV": "#64748B",
    "van": "#64748B",
    "mini-van": "#64748B",
    "bus": "#64748B",
    "vehicle": "#64748B"
}


def hex_to_bgr(hex_str: str) -> Tuple[int, int, int]:
    hex_clean = hex_str.lstrip('#')
    r = int(hex_clean[0:2], 16)
    g = int(hex_clean[2:4], 16)
    b = int(hex_clean[4:6], 16)
    return (b, g, r)


def draw_visual_annotations(image_bgr: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
    annotated = image_bgr.copy()
    h, w = annotated.shape[:2]

    # 1. Render Heavy Machinery 10-foot Exclusion Danger Zones (OSHA 1926.600)
    for det in detections:
        cls_name = det.get("class_name", "")
        if cls_name in HEAVY_EQUIPMENT:
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            radius = int(max(x2 - x1, y2 - y1) * 0.75)

            # Translucent danger perimeter
            overlay = annotated.copy()
            cv2.circle(overlay, (cx, cy), radius, (0, 140, 255), -1)
            cv2.addWeighted(overlay, 0.15, annotated, 0.85, 0, annotated)
            cv2.circle(annotated, (cx, cy), radius, (0, 165, 255), 2, cv2.LINE_AA)

    # 2. Render Object Bounding Boxes
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        cls_name = det.get("class_name", "")
        conf = det.get("confidence", 0.0)
        is_violation = cls_name in PPE_VIOLATIONS

        color_hex = CLASS_COLORS.get(cls_name, "#3B82F6")
        color_bgr = hex_to_bgr(color_hex)

        # Draw Rectangle
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color_bgr, 3 if is_violation else 2)

        # Tech corner accents
        tick = min(12, max(4, (x2 - x1) // 4), max(4, (y2 - y1) // 4))
        cv2.line(annotated, (x1, y1), (x1 + tick, y1), color_bgr, 4)
        cv2.line(annotated, (x1, y1), (x1, y1 + tick), color_bgr, 4)
        cv2.line(annotated, (x2, y1), (x2 - tick, y1), color_bgr, 4)
        cv2.line(annotated, (x2, y1), (x2, y1 + tick), color_bgr, 4)
        cv2.line(annotated, (x1, y2), (x1 + tick, y2), color_bgr, 4)
        cv2.line(annotated, (x1, y2), (x1, y2 - tick), color_bgr, 4)
        cv2.line(annotated, (x2, y2), (x2 - tick, y2), color_bgr, 4)
        cv2.line(annotated, (x2, y2), (x2, y2 - tick), color_bgr, 4)

        # Label Banner
        label = f"{'⚠️ ' if is_violation else ''}{cls_name} {int(conf * 100)}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(annotated, (x1, max(0, y1 - th - 8)), (x1 + tw + 8, y1), color_bgr, -1)
        cv2.putText(
            annotated,
            label,
            (x1 + 4, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255) if color_hex != "#EAB308" else (0, 0, 0),
            2,
            cv2.LINE_AA
        )

    return annotated


def calculate_euclidean_distance(box1: List[float], box2: List[float]) -> float:
    c1 = ((box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2)
    c2 = ((box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2)
    return math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
