"""
SiteGuard AI - Reasoning Schemas
Pydantic data contracts for Part B Natural Language Reasoning & Question Answering.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., description="Natural language question about the image or safety rules")
    conf_threshold: Optional[float] = Field(0.35, description="Confidence threshold for detection")
    iou_threshold: Optional[float] = Field(0.45, description="IoU threshold for NMS")


class GuardrailStatus(BaseModel):
    passed: bool = Field(..., description="Whether confidence and image quality guardrails passed")
    is_blurred: bool = Field(False, description="Image blur indicator")
    is_low_light: bool = Field(False, description="Low luminance indicator")
    quality_score: float = Field(..., description="Image quality score 0-100")
    confidence_gate_passed: bool = Field(..., description="Whether detections meet minimum confidence threshold")
    warning_notes: List[str] = Field(default_factory=list, description="Guardrail warnings")


class ReasoningStep(BaseModel):
    step_number: int
    component: str = Field(..., description="e.g. IntentRouter, DetectionEngine, SpatialReasoner, Guardrail")
    action: str
    result: str


class AskResponse(BaseModel):
    status: str = Field("success", description="Status code")
    question: str
    intent: str = Field(..., description="Identified intent: VISUAL_COUNT, PPE_COMPLIANCE, HAZARD_PROXIMITY, GENERAL_OSHA, OUT_OF_SCOPE")
    requires_detection: bool
    answer: str = Field(..., description="Plain-text natural language answer")
    guardrail: GuardrailStatus
    reasoning_trace: List[ReasoningStep]
    structured_data: Optional[Dict[str, Any]] = None
    detections_count: Optional[int] = 0
