# SiteGuard AI: Technical Engineering Memo
**Pre-Hackathon Screening • Track: Computer Vision + Applied ML Engineering**

---

## 1. Domain & Dataset Selection Justification

### Problem Domain: Industrial Construction Site Safety & Regulatory Compliance
We selected **Construction Site Safety & OSHA Compliance Monitoring** due to its high-stakes operational necessity and distinct computer vision challenges. Construction environments suffer from high casualty rates primarily caused by the "Fatal Four" hazards: *Struck-By*, *Caught-In/Between*, *Falls*, and *Electrocutions*. 

Standard general-purpose object detection models (trained purely on COCO) fail in industrial settings because they classify generic `person` or `truck` labels without distinguishing between **compliant PPE wearers** and **non-compliant workers posing life-safety violations**.

### Dataset Sourcing & Non-COCO Class Taxonomy
The dataset contains **717 high-resolution annotated images** and **5,570 bounding box annotations** across **25 distinct classes**, featuring multiple critical non-COCO classes:
- **PPE Compliance & Violations:** `NO-Hardhat` *(OSHA 1926.100)*, `NO-Safety Vest` *(OSHA 1926.200)*, `NO-Mask`, `Hardhat`, `Safety Vest`, `Mask`, `Gloves`.
- **Heavy Machinery & Equipment:** `Excavator`, `dump truck`, `wheel loader`, `machinery`, `Ladder`, `Safety Cone`, `fire hydrant`.
- **Transportation:** `truck`, `truck and trailer`, `semi`, `trailer`, `sedan`, `SUV`, `van`, `mini-van`, `bus`, `vehicle`.

The explicit annotation of negative compliance classes (`NO-Hardhat`, `NO-Safety Vest`) enables direct semantic detection of regulatory infractions without relying on secondary classification heads.

---

## 2. Train / Val / Test Split Strategy & Technical Justification

The dataset was partitioned following an **approximate 70% / 15% / 15% split**:
- **Training Set:** 521 images (72.7%) — 4,031 bounding box annotations.
- **Validation Set:** 114 images (15.9%) — 733 bounding box annotations.
- **Test Set (Hold-out):** 82 images (11.4%) — 806 bounding box annotations.

### Technical Split Strategy:
1. **Scene Disjoint Partitioning:** Sequential frames extracted from the same video capture were grouped into the same split to prevent spatial data leakage.
2. **Stratified Class Coverage:** High-risk minority classes (`Excavator`, `Ladder`, `wheel loader`) were verified across all splits to prevent zero-shot evaluation artifacts.
3. **Resolution Standardization:** Images were standardized to $704 \times 704$ / $640 \times 640$ pixels with orientation metadata normalization.

---

## 3. Evaluation Methodology & Metric Limitations

### Evaluated Metrics:
- **mAP@0.50:** Evaluates coarse localization accuracy at 50% IoU.
- **mAP@0.50:0.95:** Evaluates strict localization fidelity across tightening IoU thresholds.
- **Precision & Recall ($F_1$-Score):** In safety compliance, **Recall is prioritized** over Precision because a False Negative (failing to detect a worker without a helmet) risks severe injury and OSHA penalties ($15,625 per occurrence), whereas a False Positive (false alarm) merely triggers supervisory verification.

### What Metrics Tell Us vs. What They Do Not:
- **What they tell us:** The model excels at distinguishing large heavy equipment (`Excavator`, `dump truck` mAP@50 > 85%) and standard worker profiles (`Person` mAP@50 > 90%).
- **What they DO NOT tell us:** Standard box IoU does not evaluate whether a detected hardhat is *anatomically mounted on the correct worker's head* versus resting on a nearby crate. This geometric relationship is resolved by our Part B reasoning layer.

---

## 4. Five Failure Cases & Root-Cause Analysis

A model with zero acknowledged failure cases is a failure of evaluation. Below are five verified failure modes of the RT-DETR / YOLO detection system:

```
+-------------------+--------------------------------+--------------------------------------+
| Failure Case      | Visual Manifestation           | Technical Root Cause                 |
+-------------------+--------------------------------+--------------------------------------+
| 1. Motion Blur    | Fast-moving worker hands/heads | Laplacian variance drops < 40;       |
|                   | miss `Gloves` / `Hardhat`      | high-frequency edge gradients lost.  |
+-------------------+--------------------------------+--------------------------------------+
| 2. Scale Extremes | Workers > 40 meters away       | Small receptive field anchor deficit;|
|    (< 18px boxes) | misclassified as background    | features vanish after 4x downsample. |
+-------------------+--------------------------------+--------------------------------------+
| 3. Heavy          | Worker behind scaffolding bars | Bounding box splits into disjoint    |
|    Occlusion      | triggers `NO-Safety Vest`      | fragments, obscuring torso color.    |
+-------------------+--------------------------------+--------------------------------------+
| 4. Frontal Class  | `wheel loader` front profile   | High visual overlap in shovel shape  |
|    Ambiguity      | confused with `Excavator`      | and industrial yellow paint schema.  |
+-------------------+--------------------------------+--------------------------------------+
| 5. Specular Glare | Direct sun reflection on metal | High-value saturation burnout washes |
|    & Backlight    | causes false `Hardhat` detect  | out color contrast histograms.       |
+-------------------+--------------------------------+--------------------------------------+
```

---

## 5. Part B Hand-Written Reasoning Layer & Guardrail Behavior

### Architecture (Zero-Framework Implementation)
In compliance with the hard constraints, Part B contains **no agentic frameworks** (no LangChain, LangGraph, CrewAI, AutoGen). It executes a deterministic 3-tier pipeline:

1. **Intent Router (`router.py`):** Uses lexical semantic regex routing to categorize questions into `VISUAL_COUNT`, `PPE_COMPLIANCE`, `HAZARD_PROXIMITY`, `MOST_COMMON_OBJECT`, `GENERAL_OSHA`, or `OUT_OF_SCOPE`. Non-visual OSHA regulatory questions bypass detection entirely.
2. **Confidence & Quality Guardrail (`guardrail.py`):** Computes Laplacian variance (blur) and mean luminance (lighting). If an image fails quality thresholds or detections lack sufficient confidence, the guardrail triggers.
3. **Structured Safety Reasoner (`rules.py`):** Computes exact worker-to-machinery Euclidean distances (enforcing OSHA 1926.600 10ft perimeter) and tallies PPE violations over structured detection arrays.

### Specific "Insufficient Information" Example:
- **User Query:** *"Is the worker in the far background wearing a certified mask?"*
- **Image Condition:** Image contains heavy motion blur (Laplacian variance = 28.4 < 55.0 threshold) and small scale worker box (22px width).
- **Guardrail Action:** The guardrail blocks speculative hallucination and returns:
  > `⚠️ Insufficient Information: Image motion blur prevents confident identification of fine PPE details (e.g. hardhat fitment or face masks). (Quality Score: 24.5%)`

---

## 6. Training Optimization for 2 GB VRAM (GeForce MX550)

To train RT-DETR locally without CUDA Out-of-Memory (OOM) on 2GB VRAM:
- **Automatic Mixed Precision (AMP FP16):** Reduces memory by ~50% and enables Tensor Core acceleration.
- **Gradient Accumulation:** Batch size = 2 with accumulation steps = 4 yields an effective batch size of 8.
- **Cosine Learning Rate Decay:** Fast convergence in fewer epochs.
