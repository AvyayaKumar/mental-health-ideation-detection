import os
import csv
import logging
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

logger = logging.getLogger(__name__)

from worker import analyze_text, celery_app
from models.database import init_db, get_db, Feedback, FeedbackStats, Visitor, SessionLocal

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


# Visitor tracking middleware
@app.middleware("http")
async def track_visitors(request: Request, call_next):
    """Track page visits for analytics."""
    # Track page visits (but not API health checks or static assets)
    path = request.url.path
    if not path.startswith("/api/") and not path.startswith("/static") and path != "/health":
        try:
            # Parse user agent
            from user_agents import parse
            ua_string = request.headers.get("user-agent", "")
            ua = parse(ua_string)

            # Get client IP (handle proxies)
            client_ip = request.headers.get("x-forwarded-for", request.client.host if request.client else None)
            if client_ip and "," in client_ip:
                client_ip = client_ip.split(",")[0].strip()

            # Create visitor record in background (don't block request)
            db = SessionLocal()
            try:
                visitor = Visitor(
                    ip_address=client_ip,
                    user_agent=ua_string[:500],  # Truncate to fit DB
                    path=path,
                    method=request.method,
                    browser=ua.browser.family if ua.browser else None,
                    device_type=ua.device.family if ua.device else None,
                    os=ua.os.family if ua.os else None,
                )
                db.add(visitor)
                db.commit()
            finally:
                db.close()
        except Exception as e:
            # Don't let tracking errors break the app
            print(f"Visitor tracking error: {e}")

    response = await call_next(request)
    return response


# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    """Initialize database tables on application startup."""
    init_db()

    # Clean up old visitor records (keep only last 30 days)
    try:
        from datetime import timedelta
        db = SessionLocal()
        cutoff_date = datetime.utcnow() - timedelta(days=30)
        db.query(Visitor).filter(Visitor.timestamp < cutoff_date).delete()
        db.commit()
        db.close()
    except Exception as e:
        print(f"Cleanup error: {e}")


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


@app.post("/api/v1/predict")
def predict_sync(req: EssayRequest):
    """Direct synchronous prediction endpoint (no Celery required)."""
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")

    # Import pipeline here to avoid circular imports
    from models.pipeline import AnalysisPipeline
    pipeline = AnalysisPipeline()

    try:
        result = pipeline.analyze(text, explain=False)
        return result
    except Exception as e:
        import traceback
        logger.error(f"Prediction failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


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


# ============================================================================
# VISITOR ANALYTICS ENDPOINTS
# ============================================================================

@app.get("/api/v1/visitors/stats")
def get_visitor_stats(days: int = 7, db: Session = Depends(get_db)):
    """
    Get visitor statistics for the last N days.

    Query parameters:
    - days: Number of days to look back (default 7)
    """
    from datetime import timedelta

    cutoff = datetime.utcnow() - timedelta(days=days)

    # Total visitors
    total_visitors = db.query(func.count(Visitor.id)).filter(Visitor.timestamp >= cutoff).scalar()

    # Unique IPs
    unique_ips = db.query(func.count(func.distinct(Visitor.ip_address))).filter(Visitor.timestamp >= cutoff).scalar()

    # Page views by path
    page_views = db.query(
        Visitor.path,
        func.count(Visitor.id).label('count')
    ).filter(Visitor.timestamp >= cutoff).group_by(Visitor.path).order_by(func.count(Visitor.id).desc()).limit(10).all()

    # Browser breakdown
    browsers = db.query(
        Visitor.browser,
        func.count(Visitor.id).label('count')
    ).filter(Visitor.timestamp >= cutoff).group_by(Visitor.browser).order_by(func.count(Visitor.id).desc()).all()

    # Device type breakdown
    devices = db.query(
        Visitor.device_type,
        func.count(Visitor.id).label('count')
    ).filter(Visitor.timestamp >= cutoff).group_by(Visitor.device_type).order_by(func.count(Visitor.id).desc()).all()

    # OS breakdown
    operating_systems = db.query(
        Visitor.os,
        func.count(Visitor.id).label('count')
    ).filter(Visitor.timestamp >= cutoff).group_by(Visitor.os).order_by(func.count(Visitor.id).desc()).all()

    # Visits per day
    daily_visits = db.query(
        func.date(Visitor.timestamp).label('date'),
        func.count(Visitor.id).label('count')
    ).filter(Visitor.timestamp >= cutoff).group_by(func.date(Visitor.timestamp)).order_by(func.date(Visitor.timestamp)).all()

    return {
        "total_visitors": total_visitors,
        "unique_ips": unique_ips,
        "time_range_days": days,
        "page_views": [{"path": p, "count": c} for p, c in page_views],
        "browsers": [{"name": b or "Unknown", "count": c} for b, c in browsers],
        "devices": [{"type": d or "Unknown", "count": c} for d, c in devices],
        "operating_systems": [{"name": os or "Unknown", "count": c} for os, c in operating_systems],
        "daily_visits": [{"date": str(d), "count": c} for d, c in daily_visits]
    }


@app.get("/api/v1/visitors/recent")
def get_recent_visitors(limit: int = 50, db: Session = Depends(get_db)):
    """
    Get recent visitors.

    Query parameters:
    - limit: Max number of recent visitors to return (default 50)
    """
    visitors = db.query(Visitor).order_by(Visitor.timestamp.desc()).limit(limit).all()

    return {
        "count": len(visitors),
        "visitors": [v.to_dict() for v in visitors]
    }


@app.get("/admin/visitors")
def admin_visitors_page(request: Request):
    """Admin dashboard for viewing visitor analytics."""
    return templates.TemplateResponse("admin_visitors.html", {"request": request})
