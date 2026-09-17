"""FastAPI Patients Router"""
import uuid
from typing import List
from fastapi import APIRouter, HTTPException, status
from api.schemas import PatientCreate, PatientResponse, PatientDetail

router = APIRouter()
_patients_db = {}  # In-memory store (replace with DB in production)

@router.post("", response_model=PatientResponse, status_code=status.HTTP_201_CREATED)
async def create_patient(patient: PatientCreate):
    """Create a new patient record."""
    pid = uuid.uuid4()
    code = f"PAT-{str(pid)[:8].upper()}"
    record = {"id": pid, "patient_code": code, **patient.model_dump()}
    _patients_db[str(pid)] = record
    return PatientResponse(id=pid, patient_code=code,
                            first_name=patient.first_name, last_name=patient.last_name,
                            created_at=__import__("datetime").datetime.utcnow())

@router.get("/{patient_id}", response_model=PatientResponse)
async def get_patient(patient_id: uuid.UUID):
    """Get patient by ID."""
    record = _patients_db.get(str(patient_id))
    if not record:
        raise HTTPException(status_code=404, detail="Patient not found")
    return PatientResponse(**{k: v for k, v in record.items()
                               if k in PatientResponse.model_fields},
                           created_at=__import__("datetime").datetime.utcnow())

@router.get("", response_model=List[PatientResponse])
async def list_patients(skip: int = 0, limit: int = 50):
    """List all patients."""
    records = list(_patients_db.values())[skip:skip+limit]
    return [PatientResponse(id=r["id"], patient_code=r["patient_code"],
                             first_name=r["first_name"], last_name=r["last_name"],
                             created_at=__import__("datetime").datetime.utcnow())
            for r in records]
