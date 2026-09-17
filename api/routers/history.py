"""FastAPI History Router"""
import uuid
from datetime import datetime
from typing import List
from fastapi import APIRouter, HTTPException
from api.schemas import HistoryEntry, HistoryResponse

router = APIRouter()

# In-memory prediction history store
_history_db: dict = {}

def record_prediction(patient_id: uuid.UUID, prediction_data: dict):
    """Store prediction to history (called by predict router)."""
    pid = str(patient_id)
    if pid not in _history_db:
        _history_db[pid] = []
    _history_db[pid].append(prediction_data)

@router.get("/{patient_id}", response_model=HistoryResponse)
async def get_patient_history(patient_id: uuid.UUID, limit: int = 20):
    """
    Retrieve prediction history for a patient.

    Returns all predictions ordered by most recent first.
    """
    pid = str(patient_id)
    records = _history_db.get(pid, [])

    if not records:
        # Return empty history (not 404 — patient may exist but have no predictions)
        return HistoryResponse(
            patient_id=patient_id,
            patient_name="Unknown",
            total=0,
            entries=[],
        )

    entries = []
    for r in records[-limit:][::-1]:
        entries.append(HistoryEntry(
            prediction_id=r.get("prediction_id", uuid.uuid4()),
            assessed_at=r.get("assessed_at", datetime.utcnow()),
            updrs_score=r.get("updrs_score", 0.0),
            severity_label=r.get("severity_label", "Unknown"),
            confidence=r.get("confidence", 0.0),
            modalities_used=[
                m for m, avail in [
                    ("Voice", r.get("voice_available", False)),
                    ("EEG",   r.get("eeg_available",   False)),
                    ("Gait",  r.get("gait_available",  False)),
                ] if avail
            ],
        ))

    return HistoryResponse(
        patient_id=patient_id,
        patient_name=records[0].get("patient_name", "Unknown"),
        total=len(records),
        entries=entries,
    )

@router.get("/{patient_id}/trend")
async def get_updrs_trend(patient_id: uuid.UUID):
    """
    Get UPDRS score trend over time for a patient.
    Useful for tracking disease progression.
    """
    pid = str(patient_id)
    records = _history_db.get(pid, [])

    trend = [
        {
            "date":         r.get("assessed_at", datetime.utcnow()).isoformat()
                            if hasattr(r.get("assessed_at", ""), "isoformat")
                            else str(r.get("assessed_at", "")),
            "updrs_score":  r.get("updrs_score", 0.0),
            "severity":     r.get("severity_label", "Unknown"),
        }
        for r in records
    ]

    return {"patient_id": str(patient_id), "trend": trend}
