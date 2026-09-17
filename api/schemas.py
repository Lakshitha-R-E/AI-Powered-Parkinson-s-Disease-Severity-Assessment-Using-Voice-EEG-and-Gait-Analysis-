"""
============================================================
Pydantic Schemas — FastAPI Request/Response Models
AI-Powered Parkinson's Disease Severity Assessment
============================================================
"""

import uuid
from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, EmailStr, Field, validator


# ─────────────────────────────────────────────
# Common
# ─────────────────────────────────────────────

class HealthResponse(BaseModel):
    status:       str
    version:      str
    model_loaded: bool
    gpu_available:bool
    gpu_name:     Optional[str] = None


# ─────────────────────────────────────────────
# Patient Schemas
# ─────────────────────────────────────────────

class PatientCreate(BaseModel):
    first_name:           str = Field(..., min_length=1, max_length=100)
    last_name:            str = Field(..., min_length=1, max_length=100)
    date_of_birth:        Optional[datetime] = None
    gender:               Optional[str]      = None
    email:                Optional[str]      = None
    phone:                Optional[str]      = None
    address:              Optional[str]      = None
    diagnosis_date:       Optional[datetime] = None
    disease_duration_years: Optional[float] = Field(None, ge=0)
    medications:          Optional[List[str]] = []
    comorbidities:        Optional[List[str]] = []
    referring_clinician:  Optional[str]      = None
    notes:                Optional[str]      = None


class PatientResponse(BaseModel):
    id:           uuid.UUID
    patient_code: str
    first_name:   str
    last_name:    str
    age:          Optional[int] = None
    gender:       Optional[str] = None
    created_at:   datetime

    model_config = ConfigDict(from_attributes=True)


class PatientDetail(PatientResponse):
    email:                 Optional[str]      = None
    phone:                 Optional[str]      = None
    address:               Optional[str]      = None
    diagnosis_date:        Optional[datetime] = None
    disease_duration_years: Optional[float]  = None
    medications:           Optional[List[str]] = None
    comorbidities:         Optional[List[str]] = None
    referring_clinician:   Optional[str]      = None
    notes:                 Optional[str]      = None

    model_config = ConfigDict(from_attributes=True)


# ─────────────────────────────────────────────
# Prediction Schemas
# ─────────────────────────────────────────────

class PredictionRequest(BaseModel):
    """
    Request body for prediction endpoint.
    Files are sent as multipart/form-data.
    This schema handles JSON metadata alongside files.
    """
    patient_id:  Optional[uuid.UUID] = None
    notes:       Optional[str]       = None


class PredictionResult(BaseModel):
    prediction_id:       uuid.UUID
    patient_id:          Optional[uuid.UUID] = None
    updrs_score:         float = Field(..., ge=0, le=108)
    severity_class:      int   = Field(..., ge=0, le=2)
    severity_label:      str
    confidence:          float = Field(..., ge=0, le=1)
    prob_mild:           float
    prob_moderate:       float
    prob_severe:         float
    voice_available:     bool
    eeg_available:       bool
    gait_available:      bool
    inference_time_ms:   float
    model_version:       str
    assessed_at:         datetime

    model_config = ConfigDict(from_attributes=True)


class PredictionResponse(BaseModel):
    success:    bool
    message:    str
    data:       PredictionResult


# ─────────────────────────────────────────────
# Feature Schemas
# ─────────────────────────────────────────────

class VoiceFeatures(BaseModel):
    pitch_mean:        Optional[float] = None
    pitch_std:         Optional[float] = None
    jitter_local:      Optional[float] = None
    shimmer_local:     Optional[float] = None
    hnr_mean:          Optional[float] = None
    mfcc_mean:         Optional[List[float]] = None
    spectral_centroid: Optional[float] = None
    zcr:               Optional[float] = None
    rms:               Optional[float] = None


class EEGFeatures(BaseModel):
    delta_power:       Optional[float] = None
    theta_power:       Optional[float] = None
    alpha_power:       Optional[float] = None
    beta_power:        Optional[float] = None
    gamma_power:       Optional[float] = None
    spectral_entropy:  Optional[float] = None
    plv_alpha_mean:    Optional[float] = None
    plv_beta_mean:     Optional[float] = None


class GaitFeatures(BaseModel):
    cadence:           Optional[float] = None
    walking_speed:     Optional[float] = None
    step_length:       Optional[float] = None
    stride_length:     Optional[float] = None
    gait_asymmetry:    Optional[float] = None
    step_variability:  Optional[float] = None
    freeze_index:      Optional[float] = None
    swing_time:        Optional[float] = None
    stance_time:       Optional[float] = None


# ─────────────────────────────────────────────
# Report Schemas
# ─────────────────────────────────────────────

class ReportRequest(BaseModel):
    prediction_id:      uuid.UUID
    report_type:        str = "full"
    include_xai:        bool = True
    clinician_notes:    Optional[str] = None
    patient_info_override: Optional[Dict] = None


class ReportResponse(BaseModel):
    report_id:    uuid.UUID
    file_path:    str
    file_size_kb: Optional[int] = None
    generated_at: datetime
    status:       str
    download_url: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


# ─────────────────────────────────────────────
# History Schemas
# ─────────────────────────────────────────────

class HistoryEntry(BaseModel):
    prediction_id:    uuid.UUID
    assessed_at:      datetime
    updrs_score:      float
    severity_label:   str
    confidence:       float
    modalities_used:  List[str]

    model_config = ConfigDict(from_attributes=True)


class HistoryResponse(BaseModel):
    patient_id:   uuid.UUID
    patient_name: str
    total:        int
    entries:      List[HistoryEntry]


# ─────────────────────────────────────────────
# Training Schemas
# ─────────────────────────────────────────────

class TrainRequest(BaseModel):
    epochs:       int   = Field(100, ge=1,  le=1000)
    batch_size:   int   = Field(32,  ge=4,  le=256)
    learning_rate:float = Field(1e-4, ge=1e-6, le=1e-1)
    use_amp:      bool  = True
    data_path:    Optional[str] = None


class TrainResponse(BaseModel):
    job_id:       str
    status:       str
    message:      str
    estimated_time_min: Optional[float] = None
