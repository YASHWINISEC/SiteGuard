"""
SiteGuard AI - Part B Intent Routing Module (Hand-Written Heuristic Router)
Strictly adheres to Hard Constraint: NO Agentic frameworks (no LangChain, no CrewAI, no AutoGen).
Classifies natural language questions to determine if Object Detection is required.
"""

import re
from typing import Dict, Any, Tuple


class IntentRouter:
    """
    Zero-framework deterministic intent router using lexical semantics and pattern matching.
    """

    # Visual detection triggers
    VISUAL_KEYWORDS = [
        r"\bhow many\b", r"\bcount\b", r"\bnumber of\b", r"\banyone\b", r"\bwho is\b",
        r"\bwhere is\b", r"\bwhere are\b", r"\bis there\b", r"\bare there\b", r"\bdo you see\b",
        r"\bperson\b", r"\bpeople\b", r"\bworker\b", r"\bworkers\b", r"\bhelmet\b",
        r"\bhardhat\b", r"\bvest\b", r"\bmask\b", r"\bglove\b", r"\bgloves\b",
        r"\bexcavator\b", r"\btruck\b", r"\bmachinery\b", r"\bvehicle\b", r"\bladder\b",
        r"\bmost common\b", r"\bmost frequent\b", r"\bproximity\b", r"\bdanger\b",
        r"\bviolating\b", r"\bviolation\b", r"\bcompliance\b", r"\bwearing\b", r"\bmissing\b",
        r"\bwhat is in\b", r"\bwhat do you see\b", r"\blist objects\b"
    ]

    # General OSHA / Knowledge triggers (does NOT require image detection)
    GENERAL_OSHA_KEYWORDS = [
        r"\bwhat is osha\b", r"\b1926\.100\b", r"\b1926\.200\b", r"\b1926\.600\b",
        r"\b1926\.501\b", r"\bstandard penalty\b", r"\bmaximum penalty\b",
        r"\bwhat is the regulation for\b", r"\bexplain the rule\b"
    ]

    @classmethod
    def route(cls, question: str) -> Dict[str, Any]:
        """
        Routes the question into an intent category and decides if detection is required.
        """
        q_clean = question.strip().lower()

        # 1. Check for General OSHA inquiries (Non-detection visual route)
        for pattern in cls.GENERAL_OSHA_KEYWORDS:
            if re.search(pattern, q_clean):
                return {
                    "intent": "GENERAL_OSHA",
                    "requires_detection": False,
                    "target_topic": "OSHA_REGULATION",
                    "confidence": 0.95,
                    "explanation": "Question inquires about general OSHA regulatory standards and does not require analyzing specific image pixels."
                }

        # 2. Check for Specific Visual Inquiry Categories
        if re.search(r"\b(how many|count|number of)\b", q_clean):
            target = "all"
            if re.search(r"\b(person|people|worker|workers|men|man)\b", q_clean):
                target = "Person"
            elif re.search(r"\b(hardhat|helmet)\b", q_clean):
                target = "Hardhat"
            elif re.search(r"\b(vest|safety vest)\b", q_clean):
                target = "Safety Vest"
            elif re.search(r"\b(excavator|truck|machinery|vehicle)\b", q_clean):
                target = "Machinery"

            return {
                "intent": "VISUAL_COUNT",
                "requires_detection": True,
                "target_topic": target,
                "confidence": 0.98,
                "explanation": f"Visual counting query for '{target}' objects in the scene."
            }

        if re.search(r"\b(wear|wearing|without|not wearing|no-hardhat|no-vest|missing helmet|missing vest|violation|compliant)\b", q_clean):
            return {
                "intent": "PPE_COMPLIANCE",
                "requires_detection": True,
                "target_topic": "PPE_SAFETY",
                "confidence": 0.96,
                "explanation": "PPE compliance analysis required over detected workers and safety gear."
            }

        if re.search(r"\b(proximity|near|close to|danger zone|hazard|excavator distance|machinery)\b", q_clean):
            return {
                "intent": "HAZARD_PROXIMITY",
                "requires_detection": True,
                "target_topic": "EQUIPMENT_PROXIMITY",
                "confidence": 0.94,
                "explanation": "Spatial proximity and heavy equipment exclusion zone reasoning required."
            }

        if re.search(r"\b(most common|most frequent|majority|predominant)\b", q_clean):
            return {
                "intent": "MOST_COMMON_OBJECT",
                "requires_detection": True,
                "target_topic": "CLASS_FREQUENCY",
                "confidence": 0.95,
                "explanation": "Statistical frequency reasoning across detected objects required."
            }

        # Check if generic visual keyword exists
        for pattern in cls.VISUAL_KEYWORDS:
            if re.search(pattern, q_clean):
                return {
                    "intent": "GENERAL_VISUAL",
                    "requires_detection": True,
                    "target_topic": "SCENE_ANALYSIS",
                    "confidence": 0.88,
                    "explanation": "Visual detection required to analyze scene contents."
                }

        # Out-of-scope / Unrelated Question
        return {
            "intent": "OUT_OF_SCOPE",
            "requires_detection": False,
            "target_topic": "UNRELATED",
            "confidence": 0.90,
            "explanation": "Question is unrelated to construction safety or image content."
        }
