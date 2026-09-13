"""
SiteGuard AI - Part A Detection Endpoint (/api/detect)
Exposes the RT-DETR / YOLO object detection model via FastAPI.
"""

from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from typing import Optional
import cv2
import numpy as np
import base64
import time

import io
from PIL import Image

from app.schemas.detection import DetectionResponse, ObjectDetection
from app.detection.model import SiteGuardDetector

router = APIRouter(tags=["Part A - Object Detection"])
detector = SiteGuardDetector()


def decode_image_bytes(contents: bytes) -> Optional[np.ndarray]:
    """
    Decodes raw image bytes into an OpenCV BGR numpy array.
    Supports JPG, PNG, WEBP, BMP, TIFF, and exotic formats with PIL fallback.
    """
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is not None:
        return image

    try:
        pil_img = Image.open(io.BytesIO(contents))
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        img_rgb = np.array(pil_img)
        return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def encode_image_base64(img_bgr: np.ndarray) -> str:
    _, buf = cv2.imencode('.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return base64.b64encode(buf).decode('utf-8')


@router.post("/api/detect", response_model=DetectionResponse)
async def detect_objects(
    file: UploadFile = File(...),
    conf_threshold: Optional[float] = Form(0.35),
    iou_threshold: Optional[float] = Form(0.45),
    render_annotated: bool = Form(True)
):
    """
    Part A Core Endpoint:
    Accepts an image and returns detected objects, bounding boxes, confidence scores, and latency.
    """
    start_t = time.time()
    contents = await file.read()
    image = decode_image_bytes(contents)

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file or unsupported format. Please upload JPG, PNG, WEBP, or standard image format."
        )

    h, w = image.shape[:2]

    # Run inference
    detections = detector.predict(
        image_bgr=image,
        conf_threshold=conf_threshold or 0.35,
        iou_threshold=iou_threshold or 0.45
    )

    annotated_b64 = None
    if render_annotated:
        annotated_img = detector.annotate(image, detections)
        annotated_b64 = encode_image_base64(annotated_img)

    latency_ms = round((time.time() - start_t) * 1000, 2)

    return DetectionResponse(
        status="success",
        filename=file.filename,
        image_width=w,
        image_height=h,
        inference_time_ms=latency_ms,
        model_name=detector.model_name,
        detections_count=len(detections),
        detections=[ObjectDetection(**d) for d in detections],
        annotated_image_base64=annotated_b64
    )
