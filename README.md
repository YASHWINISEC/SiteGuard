# 🦺 SiteGuard AI — Constrained Object Detection & Reasoning API

![RT-DETR](https://img.shields.io/badge/Model-RT--DETR%20%7C%20YOLOv8-blue?style=for-the-badge)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?style=for-the-badge&logo=fastapi)
![React](https://img.shields.io/badge/Frontend-React%20%2B%20Vite-61DAFB?style=for-the-badge&logo=react)
![OSHA 1926](https://img.shields.io/badge/Standard-OSHA%2029%20CFR%201926-emerald?style=for-the-badge)

An end-to-end Construction Site Safety & OSHA Compliance platform featuring **Part A (RT-DETR Object Detection)** and **Part B (Hand-Written Minimal Reasoning Layer with Zero Agentic Frameworks)**.

---

## 📂 Project Architecture

```
Site/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── detect.py          # Part A: /api/detect endpoint
│   │   │   └── ask.py             # Part B: /api/ask reasoning endpoint
│   │   ├── detection/
│   │   │   ├── model.py           # RT-DETR / YOLO inference engine
│   │   │   └── postprocess.py     # NMS, box scaling & OSHA drawing
│   │   ├── reasoning/
│   │   │   ├── router.py          # Part B: Hand-written Intent Router
│   │   │   ├── rules.py           # Part B: Structured Safety Reasoner
│   │   │   └── guardrail.py       # Part B: Confidence & Blur Guardrails
│   │   ├── schemas/
│   │   │   ├── detection.py       # Pydantic schemas for Part A
│   │   │   └── reasoning.py       # Pydantic schemas for Part B
│   │   └── main.py                # FastAPI application
│   ├── models/
│   │   └── best.pt                # Checkpoint storage
│   ├── uploads/
│   ├── requirements.txt
│   ├── .env
│   └── README.md
│
├── frontend/
│   ├── src/
│   │   ├── components/            # Header, Uploader, Viewer, Q&A Box
│   │   ├── pages/Dashboard.jsx    # Interactive UI
│   │   ├── services/api.js        # API Client
│   │   ├── App.jsx, main.jsx, index.css
│   ├── package.json
│   └── vite.config.js
│
├── dataset/
│   ├── train/ (images & labels)
│   ├── val/ (images & labels)
│   ├── test/ (images & labels)
│   └── data.yaml                  # 25-class taxonomy
│
├── training/
│   ├── train.py                   # Optimized for 2GB VRAM (MX550)
│   ├── evaluate.py                # mAP, Precision, Recall, F1
│   └── README.md
│
├── MEMO.md                        # 2-Page Technical Written Memo
└── README.md
```

---

## 🚀 Quick Start Guide

### 1. Manual Model Training (Optimized for 2GB VRAM MX550)
To train the model manually with FP16 AMP and gradient accumulation:

```bash
cd training
python train.py --model rtdetr-l.pt --epochs 50 --batch 2 --accumulate 4 --imgsz 640
```

### 2. Launch FastAPI Backend
```bash
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Swagger UI available at: `http://localhost:8000/docs`.

### 3. Launch React Frontend
```bash
cd frontend
npm run dev
```
Open `http://localhost:3000` in your browser.
