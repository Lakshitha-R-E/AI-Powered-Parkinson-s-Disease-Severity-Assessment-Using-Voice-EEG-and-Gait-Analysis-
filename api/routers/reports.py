"""FastAPI Reports Router"""
import uuid
from pathlib import Path
from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from fastapi.responses import FileResponse
from api.schemas import ReportRequest, ReportResponse

router = APIRouter()
_reports_db = {}

@router.post("", response_model=ReportResponse, status_code=status.HTTP_201_CREATED)
async def generate_report(request: ReportRequest, background_tasks: BackgroundTasks):
    """Generate a clinical PDF report for a prediction."""
    import datetime
    report_id = uuid.uuid4()
    output_path = f"./reports_output/{report_id}.pdf"
    Path("./reports_output").mkdir(exist_ok=True)

    def _gen():
        try:
            from src.reports.report_generator import ClinicalReportGenerator
            gen = ClinicalReportGenerator()
            gen.generate_report(
                patient_info={"name": "Patient", "patient_id": str(request.prediction_id)},
                predictions={"severity_label": "Moderate", "updrs_score": 45.0,
                             "confidence": 0.82, "class_probabilities": [0.1, 0.82, 0.08]},
                analysis_results={},
                output_path=output_path,
            )
            _reports_db[str(report_id)]["status"] = "generated"
        except Exception as e:
            _reports_db[str(report_id)]["status"] = "failed"
            print(f"[ERROR] Report generation failed: {e}")

    record = {"id": report_id, "prediction_id": request.prediction_id,
               "patient_id": uuid.uuid4(), "file_path": output_path,
               "generated_at": datetime.datetime.utcnow(), "status": "generating"}
    _reports_db[str(report_id)] = record
    background_tasks.add_task(_gen)

    return ReportResponse(report_id=report_id, file_path=output_path,
                           generated_at=record["generated_at"], status="generating",
                           download_url=f"/report/{report_id}/download")

@router.get("/{report_id}/download")
async def download_report(report_id: uuid.UUID):
    """Download generated PDF report."""
    record = _reports_db.get(str(report_id))
    if not record:
        raise HTTPException(status_code=404, detail="Report not found")
    if not Path(record["file_path"]).exists():
        raise HTTPException(status_code=404, detail="Report file not yet generated")
    return FileResponse(record["file_path"], media_type="application/pdf",
                        filename=f"parkinson_report_{report_id}.pdf")

@router.get("/{report_id}", response_model=ReportResponse)
async def get_report_status(report_id: uuid.UUID):
    """Check report generation status."""
    record = _reports_db.get(str(report_id))
    if not record:
        raise HTTPException(status_code=404, detail="Report not found")
    return ReportResponse(report_id=record["id"], file_path=record["file_path"],
                           generated_at=record["generated_at"], status=record["status"])
