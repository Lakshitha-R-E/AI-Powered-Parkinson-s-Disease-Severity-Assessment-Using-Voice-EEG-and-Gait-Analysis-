"""
============================================================
FastAPI Router — Prediction Endpoint
AI-Powered Parkinson's Disease Severity Assessment
============================================================
POST /predict
  - Accepts voice (.wav), EEG (.edf), gait (.csv) as multipart
  - Runs processing + model inference
  - Returns UPDRS score + severity classification
============================================================
"""

import io
import time
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from api.schemas import PredictionResponse, PredictionResult, TrainRequest, TrainResponse


router = APIRouter()

# Global model (loaded once at startup)
_model = None
_device = "cpu"


def get_model():
    """Lazy-load the multimodal model."""
    global _model, _device
    if _model is None:
        import sys
        sys.path.insert(0, ".")
        from src.models.multimodal import ParkinsonMultimodalModel
        import os

        checkpoint_path = os.getenv("BEST_MODEL_PATH", "./checkpoints/best_model.pt")
        _device = "cuda" if torch.cuda.is_available() else "cpu"

        if Path(checkpoint_path).exists():
            _model = ParkinsonMultimodalModel.from_checkpoint(checkpoint_path)
        else:
            # Create default model (no weights) for demo
            _model = ParkinsonMultimodalModel(
                voice_input_dim=200,
                eeg_input_dim=150,
                gait_input_dim=40,
            )
            print("[WARN] No checkpoint found — using random weights (demo mode)")

        _model = _model.to(_device)
        _model.eval()

    return _model, _device


def process_voice_file(audio_bytes: bytes, filename: str) -> Optional[np.ndarray]:
    """Process uploaded voice file and extract features."""
    try:
        import tempfile, os
        suffix = Path(filename).suffix or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(audio_bytes)
            tmp_path = f.name

        from src.processing.voice_processor import VoiceProcessor
        proc = VoiceProcessor()
        vec  = proc.get_feature_vector(tmp_path)
        os.unlink(tmp_path)
        return vec
    except Exception as e:
        print(f"[ERROR] Voice processing failed: {e}")
        return None


def process_eeg_file(eeg_bytes: bytes, filename: str) -> Optional[np.ndarray]:
    """Process uploaded EEG file and extract features."""
    try:
        import tempfile, os
        suffix = Path(filename).suffix or ".edf"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(eeg_bytes)
            tmp_path = f.name

        from src.processing.eeg_processor import EEGProcessor
        proc = EEGProcessor(fs=256.0)
        if suffix.lower() == ".edf":
            vec = proc.get_feature_vector(tmp_path)
        else:
            # Assume numpy/csv
            data = np.load(tmp_path) if suffix == ".npy" else np.genfromtxt(tmp_path, delimiter=",")
            vec  = proc.get_feature_vector(data)
        os.unlink(tmp_path)
        return vec
    except Exception as e:
        print(f"[ERROR] EEG processing failed: {e}")
        return None


def process_gait_file(gait_bytes: bytes, filename: str) -> Optional[np.ndarray]:
    """Process uploaded gait CSV and extract features."""
    try:
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            f.write(gait_bytes)
            tmp_path = f.name

        from src.processing.gait_processor import GaitProcessor
        proc = GaitProcessor(fs=100)
        vec  = proc.get_feature_vector(tmp_path)
        os.unlink(tmp_path)
        return vec
    except Exception as e:
        print(f"[ERROR] Gait processing failed: {e}")
        return None


@router.post("", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
async def predict(
    voice_file: Optional[UploadFile] = File(None, description="Voice recording (.wav)"),
    eeg_file:   Optional[UploadFile] = File(None, description="EEG recording (.edf)"),
    gait_file:  Optional[UploadFile] = File(None, description="Gait CSV (.csv)"),
    patient_id: Optional[str]        = Form(None),
    notes:      Optional[str]        = Form(None),
):
    """
    Run multimodal Parkinson's severity assessment.

    Upload one or more signal files. Missing modalities are handled gracefully.

    - **voice_file**: .wav audio file
    - **eeg_file**:   .edf EEG file
    - **gait_file**:  .csv IMU/gait data

    Returns UPDRS score (0–108) and severity class (Mild/Moderate/Severe).
    """
    if not voice_file and not eeg_file and not gait_file:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one signal file (voice, EEG, or gait) must be provided",
        )

    t_start = time.time()
    model, device = get_model()

    # ── Feature Extraction ─────────────────────────────
    voice_vec = eeg_vec = gait_vec = None

    if voice_file:
        content   = await voice_file.read()
        voice_vec = process_voice_file(content, voice_file.filename)

    if eeg_file:
        content = await eeg_file.read()
        eeg_vec = process_eeg_file(content, eeg_file.filename)

    if gait_file:
        content  = await gait_file.read()
        gait_vec = process_gait_file(content, gait_file.filename)

    # Check if at least one succeeded
    if voice_vec is None and eeg_vec is None and gait_vec is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="All signal processing failed. Check file formats.",
        )

    # ── Pad to expected dimensions ─────────────────────
    VOICE_DIM, EEG_DIM, GAIT_DIM = 200, 150, 40

    def pad_or_truncate(vec, target_dim):
        if vec is None:
            return None
        if len(vec) < target_dim:
            return np.pad(vec, (0, target_dim - len(vec)))
        return vec[:target_dim]

    voice_vec = pad_or_truncate(voice_vec, VOICE_DIM)
    eeg_vec   = pad_or_truncate(eeg_vec,   EEG_DIM)
    gait_vec  = pad_or_truncate(gait_vec,  GAIT_DIM)

    # ── Inference ──────────────────────────────────────
    def to_tensor(vec):
        if vec is None:
            return None
        return torch.tensor(vec, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        import torch.nn.functional as F
        outputs = model(
            voice_feats=to_tensor(voice_vec),
            eeg_feats=to_tensor(eeg_vec),
            gait_feats=to_tensor(gait_vec),
        )

        updrs_score   = float(outputs["updrs_pred"][0].cpu())
        probs         = F.softmax(outputs["severity_logits"], dim=-1)[0].cpu().numpy()
        severity_class= int(np.argmax(probs))
        severity_labels= ["Mild", "Moderate", "Severe"]

    inference_ms = (time.time() - t_start) * 1000

    result = PredictionResult(
        prediction_id=uuid.uuid4(),
        patient_id=uuid.UUID(patient_id) if patient_id else None,
        updrs_score=round(updrs_score, 2),
        severity_class=severity_class,
        severity_label=severity_labels[severity_class],
        confidence=round(float(probs[severity_class]), 4),
        prob_mild=round(float(probs[0]), 4),
        prob_moderate=round(float(probs[1]), 4),
        prob_severe=round(float(probs[2]), 4),
        voice_available=voice_vec is not None,
        eeg_available=eeg_vec is not None,
        gait_available=gait_vec is not None,
        inference_time_ms=round(inference_ms, 2),
        model_version="v1.0",
        assessed_at=__import__("datetime").datetime.utcnow(),
    )

    return PredictionResponse(
        success=True,
        message=f"Assessment complete: {severity_labels[severity_class]} (UPDRS={updrs_score:.1f})",
        data=result,
    )


@router.post("/train", response_model=TrainResponse, tags=["Training"])
async def trigger_training(
    request: TrainRequest,
    background_tasks: BackgroundTasks,
):
    """
    Trigger model retraining in the background.

    Returns job ID to track training status.
    """
    job_id = str(uuid.uuid4())

    def training_job():
        import subprocess
        cmd = [
            "python", "src/training/train.py",
            "--epochs",     str(request.epochs),
            "--batch_size", str(request.batch_size),
            "--lr",         str(request.learning_rate),
        ]
        if request.data_path:
            cmd += ["--data_path", request.data_path]
        subprocess.run(cmd, check=False)

    background_tasks.add_task(training_job)

    return TrainResponse(
        job_id=job_id,
        status="started",
        message=f"Training job {job_id} started with {request.epochs} epochs",
        estimated_time_min=request.epochs * 0.5,
    )
