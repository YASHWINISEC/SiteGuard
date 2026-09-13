"""
SiteGuard AI - Structured Safety & Spatial Reasoning Rules
Performs deterministic reasoning over structured bounding boxes, classes, counts, and spatial metrics.
"""

from typing import List, Dict, Any, Tuple
from collections import Counter
import math
from app.detection.postprocess import HEAVY_EQUIPMENT, PPE_VIOLATIONS


class StructuredSafetyReasoner:
    @classmethod
    def answer_query(
        cls,
        question: str,
        intent: str,
        detections: List[Dict[str, Any]],
        image_width: int,
        image_height: int
    ) -> Dict[str, Any]:
        """
        Executes structured reasoning based on routed intent.
        """
        if intent == "GENERAL_OSHA":
            return cls._answer_general_osha(question)

        if intent == "OUT_OF_SCOPE":
            return {
                "answer": "This question does not appear related to the construction site image or safety monitoring. Please ask about personnel, PPE compliance (hard hats, vests), machinery, or safety hazards.",
                "structured_data": {"category": "out_of_scope"}
            }

        # 1. Tally detections
        class_counts = Counter([d["class_name"] for d in detections])
        persons = [d for d in detections if d["class_name"] == "Person"]
        hard_hats = [d for d in detections if d["class_name"] == "Hardhat"]
        no_hard_hats = [d for d in detections if d["class_name"] == "NO-Hardhat"]
        vests = [d for d in detections if d["class_name"] == "Safety Vest"]
        no_vests = [d for d in detections if d["class_name"] == "NO-Safety Vest"]
        machinery = [d for d in detections if d["class_name"] in HEAVY_EQUIPMENT]

        # 2. Execute Intent-Specific Logic
        if intent == "VISUAL_COUNT":
            q_lower = question.lower()
            if any(k in q_lower for k in ["person", "people", "worker"]):
                count = len(persons)
                ans = f"There {'is' if count == 1 else 'are'} {count} worker{'s' if count != 1 else ''} detected in this image."
            elif "helmet" in q_lower or "hardhat" in q_lower:
                safe_count = len(hard_hats)
                viol_count = len(no_hard_hats)
                ans = f"Detected {safe_count} compliant hardhat(s) and {viol_count} worker(s) without hardhats."
            elif "vest" in q_lower:
                safe_count = len(vests)
                viol_count = len(no_vests)
                ans = f"Detected {safe_count} compliant safety vest(s) and {viol_count} worker(s) without high-visibility vests."
            elif any(k in q_lower for k in ["excavator", "machinery", "truck"]):
                m_count = len(machinery)
                m_names = ", ".join([f"{count} {name}" for name, count in class_counts.items() if name in HEAVY_EQUIPMENT]) or "none"
                ans = f"Detected {m_count} piece(s) of heavy equipment: {m_names}."
            else:
                total = len(detections)
                details = ", ".join([f"{count} {name}" for name, count in class_counts.most_common(5)])
                ans = f"A total of {total} object(s) were detected across the scene, including: {details}."

            return {
                "answer": ans,
                "structured_data": {"counts": dict(class_counts), "total_objects": len(detections)}
            }

        if intent == "PPE_COMPLIANCE":
            total_p = len(persons)
            no_hh = len(no_hard_hats)
            no_v = len(no_vests)

            if total_p == 0:
                ans = "No personnel were detected in the frame to evaluate for PPE compliance."
            elif no_hh == 0 and no_v == 0:
                ans = f"All {total_p} detected worker(s) are fully compliant with headwear (OSHA 1926.100) and high-visibility apparel (OSHA 1926.200)."
            else:
                issues = []
                if no_hh > 0:
                    issues.append(f"{no_hh} worker(s) without hardhats (violating 29 CFR 1926.100)")
                if no_v > 0:
                    issues.append(f"{no_v} worker(s) without safety vests (violating 29 CFR 1926.200)")
                ans = f"Safety Non-Compliance Detected: Found {' and '.join(issues)} among {total_p} worker(s)."

            return {
                "answer": ans,
                "structured_data": {
                    "total_workers": total_p,
                    "no_hardhat_violations": no_hh,
                    "no_vest_violations": no_v
                }
            }

        if intent == "HAZARD_PROXIMITY":
            proximity_hazards = []
            img_diag = math.sqrt(image_width**2 + image_height**2)
            danger_dist = img_diag * 0.18

            for p_idx, p in enumerate(persons):
                for m in machinery:
                    c1 = ((p["bbox"][0] + p["bbox"][2]) / 2, (p["bbox"][1] + p["bbox"][3]) / 2)
                    c2 = ((m["bbox"][0] + m["bbox"][2]) / 2, (m["bbox"][1] + m["bbox"][3]) / 2)
                    dist = math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
                    if dist < danger_dist:
                        proximity_hazards.append(f"Worker #{p_idx+1} is in close hazard proximity to {m['class_name']}")

            if proximity_hazards:
                ans = f"CRITICAL HAZARD: {'; '.join(proximity_hazards)}. Under OSHA 29 CFR 1926.600, unauthorized workers must maintain a 10-foot clear perimeter around active machinery."
            else:
                ans = "No hazardous heavy machinery proximity breaches were observed between personnel and equipment."

            return {
                "answer": ans,
                "structured_data": {"proximity_hazards": proximity_hazards, "machinery_count": len(machinery)}
            }

        if intent == "MOST_COMMON_OBJECT":
            if not class_counts:
                ans = "No objects were detected in the image."
            else:
                top_obj, top_count = class_counts.most_common(1)[0]
                ans = f"The most common object in this image is '{top_obj}' with {top_count} instance(s) detected."
            return {
                "answer": ans,
                "structured_data": {"most_common": top_obj if class_counts else None, "frequencies": dict(class_counts)}
            }

        # Default fallback
        summary_str = ", ".join([f"{c} {n}" for n, c in class_counts.items()]) or "No objects"
        return {
            "answer": f"Scene analysis detected {len(detections)} object(s): {summary_str}.",
            "structured_data": {"counts": dict(class_counts)}
        }

    @staticmethod
    def _answer_general_osha(question: str) -> Dict[str, Any]:
        q = question.lower()
        if "1926.100" in q or "head" in q or "helmet" in q or "hardhat" in q:
            return {
                "answer": "OSHA Standard 29 CFR 1926.100 mandates that employees working in areas where there is a potential danger of head injury from impact, falling objects, or electrical shock must wear certified industrial protective helmets.",
                "structured_data": {"standard": "29 CFR 1926.100", "severity": "HIGH", "penalty": "$15,625 per occurrence"}
            }
        elif "1926.200" in q or "vest" in q or "high-vis" in q:
            return {
                "answer": "OSHA Standard 29 CFR 1926.200 requires workers exposed to traffic or operating heavy equipment to wear ANSI/ISEA 107-compliant high-visibility warning vests or garments.",
                "structured_data": {"standard": "29 CFR 1926.200", "severity": "HIGH", "penalty": "$15,625 per occurrence"}
            }
        elif "1926.600" in q or "machinery" in q or "proximity" in q or "equipment" in q:
            return {
                "answer": "OSHA Standard 29 CFR 1926.600 governs heavy construction equipment. Operators and site supervisors must enforce clear barricaded exclusion zones around equipment swing radii and pinch points.",
                "structured_data": {"standard": "29 CFR 1926.600", "severity": "CRITICAL", "penalty": "$156,259 willful/repeated"}
            }
        else:
            return {
                "answer": "OSHA 29 CFR 1926 defines safety and health regulations for construction. Key enforced standards include 1926.100 (Head Protection), 1926.200 (High-Visibility Apparel), and 1926.600 (Equipment Safety Zones).",
                "structured_data": {"standard": "OSHA 1926 Overview"}
            }
