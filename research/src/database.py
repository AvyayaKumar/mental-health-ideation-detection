"""
Database models and utilities for feedback loop system.

This module handles:
- Storing predictions with original text and model outputs
- Collecting teacher feedback/corrections
- Tracking retraining data
- Managing contact form submissions
"""

import sqlite3
from datetime import datetime
from typing import Optional, List, Dict, Any
import json
import os


class FeedbackDatabase:
    """Manages the feedback loop database for human-in-the-loop model improvement."""

    def __init__(self, db_path: str = "data/feedback.db"):
        """Initialize database connection and create tables if needed."""
        self.db_path = db_path

        # Ensure data directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        # Initialize database
        self._init_db()

    def _init_db(self):
        """Create database tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Predictions table - stores every prediction made
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    predicted_class INTEGER NOT NULL,
                    predicted_label TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    method TEXT,
                    highlighted_html TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    session_id TEXT,
                    user_agent TEXT
                )
            """)

            # Feedback table - stores teacher corrections
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prediction_id INTEGER NOT NULL,
                    correct_class INTEGER NOT NULL,
                    correct_label TEXT NOT NULL,
                    feedback_type TEXT NOT NULL,  -- 'correction', 'confirmation', 'unsure'
                    teacher_notes TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    email TEXT,
                    FOREIGN KEY (prediction_id) REFERENCES predictions(id)
                )
            """)

            # Retraining history - tracks when feedback was used for retraining
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS retraining_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    feedback_ids TEXT NOT NULL,  -- JSON array of feedback IDs used
                    num_samples INTEGER NOT NULL,
                    model_name TEXT NOT NULL,
                    training_config TEXT,  -- JSON config
                    performance_metrics TEXT,  -- JSON metrics
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    status TEXT DEFAULT 'completed'  -- 'completed', 'failed', 'in_progress'
                )
            """)

            # Contact submissions - for crash reports and general contact
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS contact_submissions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    email TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    message TEXT NOT NULL,
                    submission_type TEXT NOT NULL,  -- 'crash_report', 'bug', 'feedback', 'question'
                    error_details TEXT,  -- JSON for crash reports
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    status TEXT DEFAULT 'new',  -- 'new', 'reviewed', 'resolved'
                    response TEXT
                )
            """)

            conn.commit()

    def store_prediction(
        self,
        text: str,
        predicted_class: int,
        predicted_label: str,
        confidence: float,
        method: Optional[str] = None,
        highlighted_html: Optional[str] = None,
        session_id: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> int:
        """
        Store a prediction in the database.

        Returns:
            prediction_id: The ID of the stored prediction
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO predictions
                (text, predicted_class, predicted_label, confidence, method,
                 highlighted_html, session_id, user_agent)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (text, predicted_class, predicted_label, confidence, method,
                  highlighted_html, session_id, user_agent))
            conn.commit()
            return cursor.lastrowid

    def store_feedback(
        self,
        prediction_id: int,
        correct_class: int,
        correct_label: str,
        feedback_type: str,
        teacher_notes: Optional[str] = None,
        email: Optional[str] = None
    ) -> int:
        """
        Store teacher feedback/correction.

        Args:
            prediction_id: ID of the prediction being corrected
            correct_class: The correct class (0 or 1)
            correct_label: The correct label ("Suicidal" or "Not Suicidal")
            feedback_type: Type of feedback ('correction', 'confirmation', 'unsure')
            teacher_notes: Optional notes from teacher
            email: Optional email for follow-up

        Returns:
            feedback_id: The ID of the stored feedback
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO feedback
                (prediction_id, correct_class, correct_label, feedback_type,
                 teacher_notes, email)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (prediction_id, correct_class, correct_label, feedback_type,
                  teacher_notes, email))
            conn.commit()
            return cursor.lastrowid

    def get_feedback_for_retraining(
        self,
        feedback_type: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get feedback data for model retraining.

        Args:
            feedback_type: Filter by feedback type (e.g., 'correction')
            limit: Maximum number of samples to return

        Returns:
            List of dicts with text, correct_class, and metadata
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            query = """
                SELECT
                    p.text,
                    f.correct_class,
                    f.correct_label,
                    f.feedback_type,
                    f.teacher_notes,
                    f.timestamp,
                    f.id as feedback_id,
                    p.id as prediction_id
                FROM feedback f
                JOIN predictions p ON f.prediction_id = p.id
                WHERE 1=1
            """
            params = []

            if feedback_type:
                query += " AND f.feedback_type = ?"
                params.append(feedback_type)

            query += " ORDER BY f.timestamp DESC"

            if limit:
                query += " LIMIT ?"
                params.append(limit)

            cursor.execute(query, params)

            return [dict(row) for row in cursor.fetchall()]

    def record_retraining(
        self,
        feedback_ids: List[int],
        model_name: str,
        training_config: Dict[str, Any],
        performance_metrics: Optional[Dict[str, Any]] = None,
        status: str = 'completed'
    ) -> int:
        """
        Record a retraining event.

        Returns:
            retraining_id: The ID of the retraining record
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO retraining_history
                (feedback_ids, num_samples, model_name, training_config,
                 performance_metrics, status)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                json.dumps(feedback_ids),
                len(feedback_ids),
                model_name,
                json.dumps(training_config),
                json.dumps(performance_metrics) if performance_metrics else None,
                status
            ))
            conn.commit()
            return cursor.lastrowid

    def store_contact_submission(
        self,
        email: str,
        subject: str,
        message: str,
        submission_type: str,
        name: Optional[str] = None,
        error_details: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Store a contact form submission.

        Returns:
            submission_id: The ID of the contact submission
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO contact_submissions
                (name, email, subject, message, submission_type, error_details)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                name,
                email,
                subject,
                message,
                submission_type,
                json.dumps(error_details) if error_details else None
            ))
            conn.commit()
            return cursor.lastrowid

    def get_feedback_stats(self) -> Dict[str, Any]:
        """Get statistics about feedback collection."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Total predictions
            cursor.execute("SELECT COUNT(*) FROM predictions")
            total_predictions = cursor.fetchone()[0]

            # Total feedback
            cursor.execute("SELECT COUNT(*) FROM feedback")
            total_feedback = cursor.fetchone()[0]

            # Feedback by type
            cursor.execute("""
                SELECT feedback_type, COUNT(*) as count
                FROM feedback
                GROUP BY feedback_type
            """)
            feedback_by_type = dict(cursor.fetchall())

            # Corrections vs confirmations
            cursor.execute("""
                SELECT
                    SUM(CASE WHEN p.predicted_class != f.correct_class THEN 1 ELSE 0 END) as corrections,
                    SUM(CASE WHEN p.predicted_class = f.correct_class THEN 1 ELSE 0 END) as confirmations
                FROM feedback f
                JOIN predictions p ON f.prediction_id = p.id
            """)
            result = cursor.fetchone()
            corrections = result[0] or 0
            confirmations = result[1] or 0

            # Retraining events
            cursor.execute("SELECT COUNT(*) FROM retraining_history")
            retraining_events = cursor.fetchone()[0]

            return {
                'total_predictions': total_predictions,
                'total_feedback': total_feedback,
                'feedback_by_type': feedback_by_type,
                'corrections': corrections,
                'confirmations': confirmations,
                'retraining_events': retraining_events,
                'feedback_rate': round(total_feedback / total_predictions * 100, 2) if total_predictions > 0 else 0
            }

    def get_recent_feedback(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent feedback submissions with prediction details."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute("""
                SELECT
                    f.id,
                    f.prediction_id,
                    f.correct_class,
                    f.correct_label,
                    f.feedback_type,
                    f.teacher_notes,
                    f.email,
                    f.timestamp,
                    p.predicted_class,
                    p.predicted_label,
                    p.confidence,
                    p.text
                FROM feedback f
                JOIN predictions p ON f.prediction_id = p.id
                ORDER BY f.timestamp DESC
                LIMIT ?
            """, (limit,))

            return [dict(row) for row in cursor.fetchall()]
