# SiteGuard AI — Backend Service

FastAPI service exposing Part A (RT-DETR Object Detection) and Part B (Hand-Written Minimal Reasoning Layer).

---

## 🚀 Running the Backend

```bash
# Navigate to backend directory
cd backend

# Start the uvicorn server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive Swagger API docs available at: `http://localhost:8000/docs`.

---

## 📡 API Endpoints

### 1. Part A: `/api/detect`
- **Method:** `POST`
- **Content-Type:** `multipart/form-data`
- **Parameters:**
  - `file`: Image file (`.jpg`, `.png`)
  - `conf_threshold`: Float (default: `0.35`)
  - `iou_threshold`: Float (default: `0.45`)
  - `render_annotated`: Boolean (default: `true`)

### 2. Part B: `/api/ask`
- **Method:** `POST`
- **Content-Type:** `multipart/form-data`
- **Parameters:**
  - `question`: Natural language question (string)
  - `file`: Optional image file (required for visual questions)
