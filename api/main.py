"""
============================================================
FastAPI Application — Main Entry Point
AI-Powered Parkinson's Disease Severity Assessment
============================================================
REST API with:
  POST /predict       — Run severity prediction
  POST /train         — Trigger model retraining
  GET  /report/{id}   — Download generated report
  POST /report        — Generate report for prediction
  POST /patient       — Create patient record
  GET  /patient/{id}  — Get patient details
  GET  /history/{pid} — Get patient prediction history
  GET  /health        — Health check
  GET  /metrics       — System metrics
============================================================
"""

import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

# Add project root to path so imports work when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from api.routers import patients, predict, reports, history
from api.schemas  import HealthResponse
from src.database.models import init_engine, create_all_tables



# ─────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────

from dotenv import load_dotenv
load_dotenv()

DATABASE_URL = os.getenv(
    "ASYNC_DATABASE_URL",
    "postgresql+asyncpg://postgres:password@localhost:5432/parkinson_db"
)
MODEL_CHECKPOINT = os.getenv("BEST_MODEL_PATH", "./checkpoints/best_model.pt")


# ─────────────────────────────────────────────
# Lifespan (startup / shutdown)
# ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown logic."""
    print("🚀 Starting Parkinson's AI API...")

    # Initialize database
    try:
        init_engine(DATABASE_URL)
        await create_all_tables()
        print("✅ Database connected and tables ready")
    except Exception as e:
        print(f"⚠️  Database initialization failed: {e}")
        print("    Running without database (development mode)")

    # Create directories
    Path("./checkpoints").mkdir(exist_ok=True)
    Path("./logs").mkdir(exist_ok=True)
    Path("./reports_output").mkdir(exist_ok=True)

    print("✅ API ready")
    yield

    print("👋 Shutting down API...")


# ─────────────────────────────────────────────
# App Instance
# ─────────────────────────────────────────────

app = FastAPI(
    title="Parkinson's Disease Severity Assessment API",
    description="""
    ## AI-Powered Multimodal Parkinson's Disease Severity Assessment

    This API provides endpoints for:
    - **Prediction**: Analyze voice, EEG, and gait signals to predict UPDRS score and severity class
    - **Patients**: Manage patient records
    - **Reports**: Generate and retrieve clinical PDF reports
    - **History**: View prediction history for patients

    ### Modalities Supported
    - 🎤 **Voice**: .wav audio files
    - 🧠 **EEG**: .edf EEG files
    - 🚶 **Gait**: .csv IMU data files

    ### Severity Classes
    - 🟢 **Mild** (UPDRS 0–30)
    - 🟡 **Moderate** (UPDRS 31–60)
    - 🔴 **Severe** (UPDRS 61–108)
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


# ─────────────────────────────────────────────
# Middleware
# ─────────────────────────────────────────────

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# GZip compression for large responses
app.add_middleware(GZipMiddleware, minimum_size=1024)


# Request timing middleware
@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    start  = time.time()
    response = await call_next(request)
    elapsed  = (time.time() - start) * 1000
    response.headers["X-Response-Time-Ms"] = f"{elapsed:.2f}"
    return response


# ─────────────────────────────────────────────
# Routers
# ─────────────────────────────────────────────

app.include_router(predict.router,  prefix="/predict",  tags=["Prediction"])
app.include_router(patients.router, prefix="/patient",  tags=["Patients"])
app.include_router(reports.router,  prefix="/report",   tags=["Reports"])
app.include_router(history.router,  prefix="/history",  tags=["History"])


# ─────────────────────────────────────────────
# Root Endpoints
# ─────────────────────────────────────────────

@app.get("/", tags=["System"])
async def root():
    return {
        "message":  "Parkinson's Disease Severity Assessment API",
        "version":  "1.0.0",
        "docs":     "/docs",
        "health":   "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint for deployment monitoring."""
    import torch
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        model_loaded=Path(MODEL_CHECKPOINT).exists(),
        gpu_available=torch.cuda.is_available(),
        gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    )


@app.get("/metrics", tags=["System"])
async def system_metrics():
    """Return system resource metrics."""
    try:
        import psutil
        return {
            "cpu_percent":    psutil.cpu_percent(interval=0.5),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_gb":      psutil.virtual_memory().used / (1024**3),
            "disk_percent":   psutil.disk_usage("/").percent,
        }
    except ImportError:
        return {
            "cpu_percent": 0.0,
            "memory_percent": 0.0,
            "memory_gb": 0.0,
            "disk_percent": 0.0,
            "note": "psutil not installed"
        }


# ─────────────────────────────────────────────
# Exception Handlers
# ─────────────────────────────────────────────

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code},
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error", "detail": str(exc)},
    )


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
