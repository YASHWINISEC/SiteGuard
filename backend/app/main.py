"""
SiteGuard AI - Main FastAPI Application
Assembles Part A Detection API and Part B Hand-Written Reasoning Layer.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from app.api.detect import router as detect_router
from app.api.ask import router as ask_router

app = FastAPI(
    title="SiteGuard AI - Constrained Object Detection & Reasoning API",
    version="2.0.0",
    description="Pre-Hackathon Screening Round 1 • Construction Site Safety RT-DETR Vision & Reasoning API",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Part A and Part B Routers
app.include_router(detect_router)
app.include_router(ask_router)


@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "SiteGuard AI",
        "version": "2.0.0",
        "frameworks_used": "None (Pure Python Hand-Crafted Reasoning)",
        "model_architecture": "RT-DETR / YOLOv8"
    }


# Static upload folder mount
uploads_dir = Path(__file__).resolve().parent.parent / "uploads"
uploads_dir.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=str(uploads_dir)), name="uploads")

# Static UI mount (Web Dashboard)
static_dir = Path(__file__).resolve().parent / "static"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

