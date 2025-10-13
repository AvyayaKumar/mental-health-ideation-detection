import os
import csv
from io import StringIO
from typing import Optional, List
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from celery.result import AsyncResult
from sqlalchemy.orm import Session
from sqlalchemy import func

from worker import analyze_text, celery_app
from models.database import init_db, get_db, Feedback, FeedbackStats

ENV = os.getenv("ENV", "development")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

app = FastAPI(title="Suicide Detection Bot API", version="0.1.0")

# CORS for local testing; tighten in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*" if ENV == "development" else "http://localhost"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")


# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    """Initialize database tables on application startup."""
    init_db()


class EssayRequest(BaseModel):
    text: str


class FeedbackRequest(BaseModel):
    """Request model for submitting teacher feedback."""
    text: str = Field(..., description="The original text that was analyzed")
    original_prediction: str = Field(..., description="Model's original prediction (LOW, MEDIUM, HIGH)")
    original_score: float = Field(..., ge=0.0, le=1.0, description="Model's prediction score")
    original_confidence: float = Field(..., ge=0.0, le=1.0, description="Model's confidence")
    teacher_correction: str = Field(..., description="Teacher's correct assessment (LOW, MEDIUM, HIGH)")
    teacher_notes: Optional[str] = Field(None, description="Optional notes from teacher")
    teacher_id: Optional[str] = Field(None, description="Optional teacher identifier")
    prediction_metadata: Optional[dict] = Field(None, description="Full prediction response metadata")


class FeedbackResponse(BaseModel):
    """Response model for feedback submission."""
    id: int
    message: str
    is_correct: bool


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/v1/analyze")
def submit_essay(req: EssayRequest):
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")
    # Queue background task
    async_result = analyze_text.delay(text)
    return {"task_id": async_result.id}


@app.get("/api/v1/result/{task_id}")
def get_result(task_id: str):
    if not task_id:
        raise HTTPException(status_code=400, detail="task_id is required")
    result: AsyncResult = celery_app.AsyncResult(task_id)
    resp = {"task_id": task_id, "status": result.status}
    if result.successful():
        resp["result"] = result.get()  # result is a JSON-serializable dict
    elif result.failed():
        # Avoid leaking internal errors
        return JSONResponse(status_code=500, content={"task_id": task_id, "status": "FAILURE"})
    return resp


# ============================================================================
# FEEDBACK ENDPOINTS - Teacher Feedback Loop
# ============================================================================

@app.post("/api/v1/feedback", response_model=FeedbackResponse)
def submit_feedback(feedback: FeedbackRequest, db: Session = Depends(get_db)):
    """
    Submit teacher feedback on a model prediction.

    Teachers can report incorrect predictions to help improve the model.
    """
    # Determine if prediction was correct
    is_correct = feedback.original_prediction == feedback.teacher_correction

    # Create feedback record
    db_feedback = Feedback(
        text=feedback.text,
        original_prediction=feedback.original_prediction,
        original_score=feedback.original_score,
        original_confidence=feedback.original_confidence,
        teacher_correction=feedback.teacher_correction,
        teacher_notes=feedback.teacher_notes,
        teacher_id=feedback.teacher_id,
        prediction_metadata=feedback.prediction_metadata,
        is_correct=is_correct,
        status="pending"
    )

    db.add(db_feedback)
    db.commit()
    db.refresh(db_feedback)

    return FeedbackResponse(
        id=db_feedback.id,
        message="Thank you for your feedback! This will help improve the model." if not is_correct else "Thank you for confirming the prediction!",
        is_correct=is_correct
    )


@app.get("/api/v1/feedback")
def list_feedback(
    skip: int = 0,
    limit: int = 100,
    status: Optional[str] = None,
    is_correct: Optional[bool] = None,
    db: Session = Depends(get_db)
):
    """
    List all feedback submissions (admin endpoint).

    Query parameters:
    - skip: Number of records to skip (pagination)
    - limit: Maximum records to return (default 100)
    - status: Filter by status (pending, reviewed, incorporated)
    - is_correct: Filter by correctness (true/false)
    """
    query = db.query(Feedback)

    if status:
        query = query.filter(Feedback.status == status)

    if is_correct is not None:
        query = query.filter(Feedback.is_correct == is_correct)

    # Order by most recent first
    query = query.order_by(Feedback.created_at.desc())

    total = query.count()
    feedbacks = query.offset(skip).limit(limit).all()

    return {
        "total": total,
        "skip": skip,
        "limit": limit,
        "data": [f.to_dict() for f in feedbacks]
    }


@app.get("/api/v1/feedback/stats")
def get_feedback_stats(db: Session = Depends(get_db)):
    """
    Get aggregate statistics on model performance based on feedback.

    Returns overall accuracy and per-class breakdown.
    """
    total = db.query(func.count(Feedback.id)).scalar()
    correct = db.query(func.count(Feedback.id)).filter(Feedback.is_correct == True).scalar()
    incorrect = db.query(func.count(Feedback.id)).filter(Feedback.is_correct == False).scalar()

    # Per-class stats
    low_correct = db.query(func.count(Feedback.id)).filter(
        Feedback.original_prediction == "LOW",
        Feedback.is_correct == True
    ).scalar()

    low_incorrect = db.query(func.count(Feedback.id)).filter(
        Feedback.original_prediction == "LOW",
        Feedback.is_correct == False
    ).scalar()

    medium_correct = db.query(func.count(Feedback.id)).filter(
        Feedback.original_prediction == "MEDIUM",
        Feedback.is_correct == True
    ).scalar()

    medium_incorrect = db.query(func.count(Feedback.id)).filter(
        Feedback.original_prediction == "MEDIUM",
        Feedback.is_correct == False
    ).scalar()

    high_correct = db.query(func.count(Feedback.id)).filter(
        Feedback.original_prediction == "HIGH",
        Feedback.is_correct == True
    ).scalar()

    high_incorrect = db.query(func.count(Feedback.id)).filter(
        Feedback.original_prediction == "HIGH",
        Feedback.is_correct == False
    ).scalar()

    accuracy = (correct / total * 100) if total > 0 else 0

    return {
        "total_feedback": total,
        "correct_predictions": correct,
        "incorrect_predictions": incorrect,
        "accuracy_rate": round(accuracy, 2),
        "class_breakdown": {
            "LOW": {
                "correct": low_correct,
                "incorrect": low_incorrect,
                "total": low_correct + low_incorrect,
                "accuracy": round((low_correct / (low_correct + low_incorrect) * 100), 2) if (low_correct + low_incorrect) > 0 else 0
            },
            "MEDIUM": {
                "correct": medium_correct,
                "incorrect": medium_incorrect,
                "total": medium_correct + medium_incorrect,
                "accuracy": round((medium_correct / (medium_correct + medium_incorrect) * 100), 2) if (medium_correct + medium_incorrect) > 0 else 0
            },
            "HIGH": {
                "correct": high_correct,
                "incorrect": high_incorrect,
                "total": high_correct + high_incorrect,
                "accuracy": round((high_correct / (high_correct + high_incorrect) * 100), 2) if (high_correct + high_incorrect) > 0 else 0
            }
        }
    }


@app.get("/api/v1/feedback/export")
def export_feedback(
    format: str = "csv",
    status: Optional[str] = None,
    is_correct: Optional[bool] = None,
    db: Session = Depends(get_db)
):
    """
    Export feedback data for model retraining.

    Query parameters:
    - format: Export format (csv, jsonl) - default csv
    - status: Filter by status
    - is_correct: Filter by correctness

    Returns a downloadable file with feedback data.
    """
    query = db.query(Feedback)

    if status:
        query = query.filter(Feedback.status == status)

    if is_correct is not None:
        query = query.filter(Feedback.is_correct == is_correct)

    feedbacks = query.order_by(Feedback.created_at.desc()).all()

    if format == "csv":
        # Create CSV
        output = StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow([
            "id", "text", "original_prediction", "original_score",
            "teacher_correction", "is_correct", "teacher_notes",
            "created_at", "status"
        ])

        # Data rows
        for f in feedbacks:
            writer.writerow([
                f.id,
                f.text,
                f.original_prediction,
                f.original_score,
                f.teacher_correction,
                f.is_correct,
                f.teacher_notes or "",
                f.created_at.isoformat(),
                f.status
            ])

        output.seek(0)
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=feedback_export_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"}
        )

    elif format == "jsonl":
        # Create JSONL (JSON Lines) for ML training
        import json

        output = StringIO()
        for f in feedbacks:
            record = {
                "text": f.text,
                "label": f.teacher_correction,
                "original_prediction": f.original_prediction,
                "is_correct": f.is_correct,
                "metadata": {
                    "id": f.id,
                    "original_score": f.original_score,
                    "teacher_notes": f.teacher_notes,
                    "created_at": f.created_at.isoformat()
                }
            }
            output.write(json.dumps(record) + "\n")

        output.seek(0)
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="application/x-ndjson",
            headers={"Content-Disposition": f"attachment; filename=feedback_export_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.jsonl"}
        )

    else:
        raise HTTPException(status_code=400, detail="Invalid format. Use 'csv' or 'jsonl'.")


@app.get("/admin/feedback")
def admin_feedback_page(request: Request):
    """Admin dashboard for reviewing feedback."""
    return templates.TemplateResponse("admin_feedback.html", {"request": request})
