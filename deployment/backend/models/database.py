"""
Database models for feedback storage.

This module defines the SQLAlchemy models for storing teacher feedback
on model predictions, enabling continuous improvement through human-in-the-loop.
"""
import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Text, Float, DateTime, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Database URL from environment or default to SQLite
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./feedback.db")

# Create engine
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


class Feedback(Base):
    """
    Teacher feedback on model predictions.

    This table stores corrections and feedback from teachers when the model
    makes incorrect predictions, enabling model retraining and improvement.
    """
    __tablename__ = "feedback"

    # Primary key
    id = Column(Integer, primary_key=True, index=True)

    # Original text and prediction
    text = Column(Text, nullable=False)  # The original essay/text analyzed
    original_prediction = Column(String(20), nullable=False)  # LOW, MEDIUM, HIGH
    original_score = Column(Float, nullable=False)  # Model confidence score (0-1)
    original_confidence = Column(Float, nullable=False)  # Model confidence (0-1)

    # Teacher correction
    teacher_correction = Column(String(20), nullable=False)  # Teacher's assessment: LOW, MEDIUM, HIGH
    teacher_notes = Column(Text, nullable=True)  # Optional notes from teacher
    teacher_id = Column(String(100), nullable=True)  # Optional teacher identifier

    # Metadata
    prediction_metadata = Column(JSON, nullable=True)  # Full prediction response (triggers, guidance, etc.)

    # Agreement/disagreement tracking
    is_correct = Column(Boolean, nullable=False, default=False)  # True if teacher agrees with model

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    reviewed_at = Column(DateTime, nullable=True)  # When feedback was reviewed by admin

    # Status tracking
    status = Column(String(20), default="pending", nullable=False)  # pending, reviewed, incorporated
    used_for_training = Column(Boolean, default=False, nullable=False)  # Whether used in retraining

    def to_dict(self):
        """Convert feedback to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "text": self.text,
            "original_prediction": self.original_prediction,
            "original_score": self.original_score,
            "original_confidence": self.original_confidence,
            "teacher_correction": self.teacher_correction,
            "teacher_notes": self.teacher_notes,
            "teacher_id": self.teacher_id,
            "is_correct": self.is_correct,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "reviewed_at": self.reviewed_at.isoformat() if self.reviewed_at else None,
            "status": self.status,
            "used_for_training": self.used_for_training,
            "prediction_metadata": self.prediction_metadata,
        }


class FeedbackStats(Base):
    """
    Aggregate statistics on model performance based on feedback.

    This table tracks overall accuracy and performance metrics over time.
    """
    __tablename__ = "feedback_stats"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Accuracy metrics
    total_feedback = Column(Integer, default=0)
    correct_predictions = Column(Integer, default=0)
    incorrect_predictions = Column(Integer, default=0)
    accuracy_rate = Column(Float, default=0.0)  # correct / total

    # Per-class metrics
    low_correct = Column(Integer, default=0)
    low_incorrect = Column(Integer, default=0)
    medium_correct = Column(Integer, default=0)
    medium_incorrect = Column(Integer, default=0)
    high_correct = Column(Integer, default=0)
    high_incorrect = Column(Integer, default=0)

    def to_dict(self):
        """Convert stats to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "date": self.date.isoformat() if self.date else None,
            "total_feedback": self.total_feedback,
            "correct_predictions": self.correct_predictions,
            "incorrect_predictions": self.incorrect_predictions,
            "accuracy_rate": self.accuracy_rate,
            "class_breakdown": {
                "low": {"correct": self.low_correct, "incorrect": self.low_incorrect},
                "medium": {"correct": self.medium_correct, "incorrect": self.medium_incorrect},
                "high": {"correct": self.high_correct, "incorrect": self.high_incorrect},
            }
        }


class Visitor(Base):
    """
    Track website visitors for basic analytics.

    Keeps recent visitor data for fun analytics. Old records can be
    automatically purged to save space.
    """
    __tablename__ = "visitors"

    id = Column(Integer, primary_key=True, index=True)

    # Visit details
    ip_address = Column(String(50), nullable=True)  # Can be anonymized
    user_agent = Column(String(500), nullable=True)  # Browser info
    path = Column(String(200), nullable=False)  # Page visited
    method = Column(String(10), nullable=False)  # GET, POST, etc.

    # Geographic/device info (parsed from user agent)
    browser = Column(String(50), nullable=True)
    device_type = Column(String(50), nullable=True)  # desktop, mobile, tablet
    os = Column(String(50), nullable=True)

    # Timing
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    # Optional: track if this led to an analysis
    led_to_analysis = Column(Boolean, default=False)

    def to_dict(self):
        """Convert visitor to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "ip_address": self.anonymize_ip(self.ip_address) if self.ip_address else None,
            "path": self.path,
            "method": self.method,
            "browser": self.browser,
            "device_type": self.device_type,
            "os": self.os,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "led_to_analysis": self.led_to_analysis,
        }

    @staticmethod
    def anonymize_ip(ip: str) -> str:
        """Anonymize IP address for privacy (keeps first 3 octets)."""
        if not ip:
            return None
        parts = ip.split('.')
        if len(parts) == 4:
            return f"{parts[0]}.{parts[1]}.{parts[2]}.xxx"
        return "xxx.xxx.xxx.xxx"


def init_db():
    """Initialize database tables."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Dependency for getting database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
