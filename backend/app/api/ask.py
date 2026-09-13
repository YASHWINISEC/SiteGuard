"""
SiteGuard AI - Part B Reasoning Endpoint (/api/ask)
Natural language question answering with Hand-Written Intent Routing, Structured Reasoning, and Confidence Guardrails.
Strictly Zero Agentic Frameworks.
"""

from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from typing import Optional, List
import cv2
import numpy as np
import io
from PIL import Image

from app.schemas.reasoning import AskResponse, GuardrailStatus, ReasoningStep
from app.detection.model import SiteGuardDetector
from app.reasoning.router import IntentRouter
from app.reasoning.rules import StructuredSafetyReasoner
from app.reasoning.guardrail import ConfidenceGuardrail

router = APIRouter(tags=["Part B - Reasoning Layer"])
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


@router.post("/api/ask", response_model=AskResponse)
async def ask_question_about_image(
    question: str = Form(..., description="Natural language question"),
    file: Optional[UploadFile] = File(None, description="Image context (required for visual questions)"),
    conf_threshold: Optional[float] = Form(0.35),
    iou_threshold: Optional[float] = Form(0.45)
):
    """
    Part B Core Endpoint:
    1. Intent Routing: Decides if detection model is needed or not.
    2. Structured Reasoning: Calls Part A model if needed and computes answer over bounding boxes/counts.
    3. Confidence Guardrail: Evaluates certainty; outputs 'Insufficient information' when uncertain.
    """
    reasoning_trace: List[ReasoningStep] = []

    # Step 1: Intent Routing
    route_info = IntentRouter.route(question)
    intent = route_info["intent"]
    requires_detection = route_info["requires_detection"]

    reasoning_trace.append(ReasoningStep(
        step_number=1,
        component="IntentRouter",
        action=f"Analyzed question: '{question}'",
        result=f"Classified Intent='{intent}', RequiresDetection={requires_detection}. {route_info['explanation']}"
    ))

    # Step 2: Handle non-detection questions
    if not requires_detection:
        reasoning_res = StructuredSafetyReasoner.answer_query(
            question=question,
            intent=intent,
            detections=[],
            image_width=0,
            image_height=0
        )
        reasoning_trace.append(ReasoningStep(
            step_number=2,
            component="KnowledgeReasoner",
            action="Executed non-visual safety standard lookup",
            result="Generated answer directly without vision inference."
        ))

        return AskResponse(
            status="success",
            question=question,
            intent=intent,
            requires_detection=False,
            answer=reasoning_res["answer"],
            guardrail=GuardrailStatus(
                passed=True,
                is_blurred=False,
                is_low_light=False,
                quality_score=100.0,
                confidence_gate_passed=True,
                warning_notes=[]
            ),
            reasoning_trace=reasoning_trace,
            structured_data=reasoning_res.get("structured_data"),
            detections_count=0
        )

    # Step 3: Decode Visual Input
    if file is None:
        raise HTTPException(
            status_code=400,
            detail="Visual questions require an uploaded image file."
        )

    contents = await file.read()
    image = decode_image_bytes(contents)

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file or unsupported format. Please upload JPG, PNG, WEBP, or standard image format."
        )

    h, w = image.shape[:2]

    # Step 4: Run Part A Object Detection Model
    detections = detector.predict(
        image_bgr=image,
        conf_threshold=conf_threshold or 0.35,
        iou_threshold=iou_threshold or 0.45
    )

    reasoning_trace.append(ReasoningStep(
        step_number=2,
        component="DetectionEngine (RT-DETR/YOLO)",
        action=f"Ran vision inference on {w}x{h} frame",
        result=f"Extracted {len(detections)} bounding boxes with conf >= {conf_threshold}."
    ))

    # Step 5: Confidence & Quality Guardrail Check
    guardrail_result = ConfidenceGuardrail.evaluate(
        image_bgr=image,
        detections=detections,
        question=question,
        intent=intent
    )

    reasoning_trace.append(ReasoningStep(
        step_number=3,
        component="ConfidenceGuardrail",
        action="Evaluated blur, lighting, and detection certainty",
        result=f"GuardrailPassed={guardrail_result['passed']}, QualityScore={guardrail_result['quality_score']}%."
    ))

    # If Guardrail fails, honestly report insufficient information
    if not guardrail_result["passed"]:
        reason_msg = guardrail_result.get("insufficient_reason") or "Insufficient visual confidence to answer accurately."
        answer = f"⚠️ Insufficient Information: {reason_msg}"

        reasoning_trace.append(ReasoningStep(
            step_number=4,
            component="GuardrailIntervention",
            action="Blocked speculative hallucination",
            result=f"Returned honest 'Insufficient Information' message: {reason_msg}"
        ))

        return AskResponse(
            status="insufficient_information",
            question=question,
            intent=intent,
            requires_detection=True,
            answer=answer,
            guardrail=GuardrailStatus(**guardrail_result),
            reasoning_trace=reasoning_trace,
            structured_data={"guardrail_intervened": True, "reason": reason_msg},
            detections_count=len(detections)
        )

    # Step 6: Structured Reasoning over Detections
    reasoning_res = StructuredSafetyReasoner.answer_query(
        question=question,
        intent=intent,
        detections=detections,
        image_width=w,
        image_height=h
    )

    reasoning_trace.append(ReasoningStep(
        step_number=4,
        component="StructuredSafetyReasoner",
        action=f"Reasoned over {len(detections)} structured boxes for intent '{intent}'",
        result="Generated plain-language synthesized answer."
    ))

    return AskResponse(
        status="success",
        question=question,
        intent=intent,
        requires_detection=True,
        answer=reasoning_res["answer"],
        guardrail=GuardrailStatus(**guardrail_result),
        reasoning_trace=reasoning_trace,
        structured_data=reasoning_res.get("structured_data"),
        detections_count=len(detections)
    )
