"""
SiteGuard AI - Detection Schemas
Pydantic data contracts for Part A Object Detection API.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x1: float = Field(..., description="Top-left X coordinate in pixels")
    y1: float = Field(..., description="Top-left Y coordinate in pixels")
    x2: float = Field(..., description="Bottom-right X coordinate in pixels")
    y2: float = Field(..., description="Bottom-right Y coordinate in pixels")


class ObjectDetection(BaseModel):
    class_id: int = Field(..., description="Numerical class index")
    class_name: str = Field(..., description="Class name label (e.g. Hardhat, NO-Hardhat, Excavator)")
    confidence: float = Field(..., description="Prediction confidence score between 0.0 and 1.0")
    bbox: List[float] = Field(..., description="Bounding box [x1, y1, x2, y2]")
    color: Optional[str] = Field("#3B82F6", description="Hex color for visualization")
    is_violation: bool = Field(False, description="Whether object represents a safety violation")


class DetectionResponse(BaseModel):
    status: str = Field("success", description="Status code")
    filename: Optional[str] = None
    image_width: int
    image_height: int
    inference_time_ms: float
    model_name: str
    detections_count: int
    detections: List[ObjectDetection]
    annotated_image_base64: Optional[str] = None
