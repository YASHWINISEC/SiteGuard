"""
SiteGuard AI - Object Detection Model Inference Engine
Handles RT-DETR and YOLOv8 model loading, inference execution, and dynamic fallback.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import cv2
import numpy as np
import torch
from ultralytics import YOLO

from app.detection.postprocess import CLASSES, CLASS_COLORS, PPE_VIOLATIONS, draw_visual_annotations


class SiteGuardDetector:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(SiteGuardDetector, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self, weights_path: Optional[str] = None):
        if self.initialized:
            return

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.weights_path = weights_path or self._resolve_weights_path()
        self.model = None
        self.is_custom_trained = False
        self.model_name = "SiteGuard Safety Engine"
        self.load_model()
        self.initialized = True

    def _resolve_weights_path(self) -> str:
        base_dir = Path(__file__).resolve().parent.parent.parent
        candidates = [
            base_dir / "models" / "best.pt",
            base_dir.parent / "training" / "runs" / "train" / "siteguard_rtdetr" / "weights" / "best.pt"
        ]
        for c in candidates:
            if c.exists() and c.stat().st_size > 10000:
                return str(c)
        return str(base_dir / "models" / "best.pt")

    def load_model(self):
        """Loads custom trained model if present, or initializes base fallback."""
        if os.path.exists(self.weights_path) and os.path.getsize(self.weights_path) > 10000:
            try:
                print(f"[SiteGuard] Loading custom checkpoint: {self.weights_path}")
                self.model = YOLO(self.weights_path)
                self.is_custom_trained = True
                self.model_name = f"SiteGuard Custom YOLO ({Path(self.weights_path).name})"
                return
            except Exception as e:
                print(f"[SiteGuard] Warning loading {self.weights_path}: {e}")

        # Fallback to base model for immediate out-of-the-box operation
        print("[SiteGuard] Loading base model fallback with PPE simulation...")
        try:
            self.model = YOLO("yolov8s.pt")
            self.model_name = "YOLOv8s Base (Demo Safety Engine)"
        except Exception:
            self.model = YOLO("yolov8n.pt")
            self.model_name = "YOLOv8n Base (Demo Safety Engine)"
        self.is_custom_trained = False

    def predict(
        self,
        image_bgr: np.ndarray,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        img_size: int = 640
    ) -> List[Dict[str, Any]]:
        if self.model is None or image_bgr is None:
            return []

        h, w = image_bgr.shape[:2]

        results = self.model.predict(
            source=image_bgr,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=img_size,
            device=self.device,
            verbose=False
        )

        detections = []
        if len(results) == 0 or results[0].boxes is None:
            return detections

        r = results[0]
        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)

        if self.is_custom_trained:
            # Custom 25-class mapping from trained model
            for box, score, cls_id in zip(boxes, scores, classes):
                cls_name = r.names.get(cls_id, CLASSES[cls_id] if cls_id < len(CLASSES) else f"class_{cls_id}")
                x1, y1, x2, y2 = [round(float(v), 1) for v in box]
                detections.append({
                    "class_id": int(cls_id),
                    "class_name": cls_name,
                    "confidence": round(float(score), 3),
                    "bbox": [x1, y1, x2, y2],
                    "color": CLASS_COLORS.get(cls_name, "#3B82F6"),
                    "is_violation": cls_name in PPE_VIOLATIONS
                })
        else:
            # Smart Fallback Simulation on base model detections
            for box, score, cls_id in zip(boxes, scores, classes):
                coco_name = r.names.get(cls_id, "")
                x1, y1, x2, y2 = [round(float(v), 1) for v in box]
                bw, bh = x2 - x1, y2 - y1

                if coco_name == "person":
                    detections.append({
                        "class_id": 8,
                        "class_name": "Person",
                        "confidence": round(float(score), 3),
                        "bbox": [x1, y1, x2, y2],
                        "color": "#3B82F6",
                        "is_violation": False
                    })
                    # Head region check
                    head_y2 = y1 + bh * 0.22
                    head_crop = image_bgr[max(0, int(y1)):max(0, int(head_y2)), max(0, int(x1)):max(0, int(x2))]
                    has_hat = self._detect_color_presence(head_crop, ["yellow", "white", "orange", "blue"])
                    
                    if has_hat:
                        detections.append({
                            "class_id": 2,
                            "class_name": "Hardhat",
                            "confidence": round(min(0.98, float(score) * 0.95), 3),
                            "bbox": [x1, y1, x2, round(head_y2, 1)],
                            "color": CLASS_COLORS["Hardhat"],
                            "is_violation": False
                        })
                    else:
                        detections.append({
                            "class_id": 5,
                            "class_name": "NO-Hardhat",
                            "confidence": round(min(0.95, float(score) * 0.92), 3),
                            "bbox": [x1, y1, x2, round(head_y2, 1)],
                            "color": CLASS_COLORS["NO-Hardhat"],
                            "is_violation": True
                        })

                    # Torso vest check
                    torso_y1 = y1 + bh * 0.25
                    torso_y2 = y1 + bh * 0.65
                    torso_crop = image_bgr[max(0, int(torso_y1)):max(0, int(torso_y2)), max(0, int(x1)):max(0, int(x2))]
                    has_vest = self._detect_color_presence(torso_crop, ["neon_yellow", "orange", "bright_green"])

                    if has_vest:
                        detections.append({
                            "class_id": 11,
                            "class_name": "Safety Vest",
                            "confidence": round(min(0.96, float(score) * 0.94), 3),
                            "bbox": [x1, round(torso_y1, 1), x2, round(torso_y2, 1)],
                            "color": CLASS_COLORS["Safety Vest"],
                            "is_violation": False
                        })
                    else:
                        detections.append({
                            "class_id": 7,
                            "class_name": "NO-Safety Vest",
                            "confidence": round(min(0.93, float(score) * 0.89), 3),
                            "bbox": [x1, round(torso_y1, 1), x2, round(torso_y2, 1)],
                            "color": CLASS_COLORS["NO-Safety Vest"],
                            "is_violation": True
                        })

                elif coco_name in ["truck", "bus"]:
                    eq_name = "Excavator" if bw > bh * 1.3 else "dump truck"
                    detections.append({
                        "class_id": 0 if eq_name == "Excavator" else 13,
                        "class_name": eq_name,
                        "confidence": round(float(score), 3),
                        "bbox": [x1, y1, x2, y2],
                        "color": CLASS_COLORS.get(eq_name, "#F97316"),
                        "is_violation": False
                    })
                elif coco_name == "car":
                    detections.append({
                        "class_id": 17,
                        "class_name": "sedan",
                        "confidence": round(float(score), 3),
                        "bbox": [x1, y1, x2, y2],
                        "color": CLASS_COLORS.get("sedan", "#64748B"),
                        "is_violation": False
                    })

        return detections

    def annotate(self, image_bgr: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        return draw_visual_annotations(image_bgr, detections)

    @staticmethod
    def _detect_color_presence(crop: np.ndarray, colors: List[str]) -> bool:
        if crop is None or crop.size < 40:
            return False
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        masks = []
        if "yellow" in colors or "neon_yellow" in colors:
            masks.append(cv2.inRange(hsv, np.array([18, 90, 90]), np.array([36, 255, 255])))
        if "orange" in colors:
            masks.append(cv2.inRange(hsv, np.array([8, 110, 110]), np.array([22, 255, 255])))
        if "bright_green" in colors:
            masks.append(cv2.inRange(hsv, np.array([35, 75, 75]), np.array([85, 255, 255])))
        if "white" in colors:
            masks.append(cv2.inRange(hsv, np.array([0, 0, 175]), np.array([180, 40, 255])))
        if "blue" in colors:
            masks.append(cv2.inRange(hsv, np.array([90, 75, 75]), np.array([130, 255, 255])))

        total = crop.shape[0] * crop.shape[1]
        for m in masks:
            if (cv2.countNonZero(m) / total) > 0.14:
                return True
        return False
