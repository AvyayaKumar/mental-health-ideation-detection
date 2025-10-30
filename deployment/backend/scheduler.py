"""
Background scheduler for automatic feedback monitoring.

This runs inside the FastAPI app and automatically:
- Checks feedback status daily
- Exports corrections when threshold is reached
- Logs status for monitoring

No manual intervention needed - fully automated!
"""

import os
import logging
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from sqlalchemy.orm import Session
from sqlalchemy import func

logger = logging.getLogger(__name__)


class FeedbackMonitor:
    """Automatic feedback monitoring and export."""

    def __init__(self, db_session_factory, min_corrections: int = 50):
        """
        Initialize feedback monitor.

        Args:
            db_session_factory: SQLAlchemy session factory
            min_corrections: Minimum incorrect predictions needed for export
        """
        self.session_factory = db_session_factory
        self.min_corrections = min_corrections
        self.last_export_count = 0
        self.scheduler = BackgroundScheduler()

    def check_and_export(self):
        """Check feedback status and auto-export if ready."""
        try:
            from models.database import Feedback

            db = self.session_factory()

            try:
                # Count incorrect predictions
                incorrect_count = db.query(func.count(Feedback.id)).filter(
                    Feedback.is_correct == False
                ).scalar()

                total_count = db.query(func.count(Feedback.id)).scalar()

                logger.info(f"📊 Feedback check: {incorrect_count} corrections / {total_count} total")

                # Check if we've reached the threshold AND haven't exported this batch yet
                if incorrect_count >= self.min_corrections and incorrect_count > self.last_export_count:
                    logger.info(f"✅ Threshold reached! Auto-exporting {incorrect_count} corrections...")

                    # Export to CSV
                    export_path = self._export_corrections(db, incorrect_count)

                    if export_path:
                        self.last_export_count = incorrect_count
                        logger.info(f"✅ Auto-export complete: {export_path}")
                        logger.info("📧 Ready to retrain! Check logs for export file location.")
                    else:
                        logger.error("❌ Auto-export failed")

                elif incorrect_count >= self.min_corrections:
                    logger.info(f"✅ Ready to retrain (already exported {self.last_export_count} corrections)")

                else:
                    needed = self.min_corrections - incorrect_count
                    logger.info(f"📊 Status: Need {needed} more corrections before auto-export")

            finally:
                db.close()

        except Exception as e:
            logger.error(f"❌ Error in feedback monitor: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def _export_corrections(self, db, count: int) -> str:
        """Export corrections to CSV file."""
        try:
            from models.database import Feedback
            import csv
            from io import StringIO

            # Query incorrect predictions
            feedbacks = db.query(Feedback).filter(
                Feedback.is_correct == False
            ).order_by(Feedback.created_at.desc()).all()

            if not feedbacks:
                logger.warning("No corrections found to export")
                return None

            # Create export directory
            export_dir = os.getenv("EXPORT_DIR", "/tmp/feedback_exports")
            os.makedirs(export_dir, exist_ok=True)

            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"corrections_{count}samples_{timestamp}.csv"
            filepath = os.path.join(export_dir, filename)

            # Write CSV
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)

                # Header
                writer.writerow(['text', 'class'])

                # Data - map risk levels to binary classification
                for fb in feedbacks:
                    # Map HIGH/MEDIUM -> suicide, LOW -> non-suicide
                    if fb.teacher_correction in ['HIGH', 'MEDIUM']:
                        label = 'suicide'
                    else:
                        label = 'non-suicide'

                    writer.writerow([fb.text, label])

            logger.info(f"✅ Exported {len(feedbacks)} corrections to: {filepath}")

            # Also create metadata file
            metadata_path = filepath.replace('.csv', '_metadata.txt')
            with open(metadata_path, 'w') as f:
                f.write(f"Feedback Export Metadata\n")
                f.write(f"========================\n\n")
                f.write(f"Export Date: {datetime.now().isoformat()}\n")
                f.write(f"Total Corrections: {len(feedbacks)}\n")
                f.write(f"File: {filename}\n\n")
                f.write(f"Class Distribution:\n")

                # Count classes
                class_counts = {}
                for fb in feedbacks:
                    label = 'suicide' if fb.teacher_correction in ['HIGH', 'MEDIUM'] else 'non-suicide'
                    class_counts[label] = class_counts.get(label, 0) + 1

                for label, count in class_counts.items():
                    f.write(f"  {label}: {count}\n")

                f.write(f"\nNext Steps:\n")
                f.write(f"1. Download file from: {filepath}\n")
                f.write(f"2. Run retraining script:\n")
                f.write(f"   python scripts/retrain_model.py --feedback-data {filename}\n")

            logger.info(f"✅ Metadata saved to: {metadata_path}")

            return filepath

        except Exception as e:
            logger.error(f"❌ Export failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def start(self):
        """Start the background scheduler."""
        # Run check daily at 9 AM
        self.scheduler.add_job(
            self.check_and_export,
            CronTrigger(hour=9, minute=0),
            id='daily_feedback_check',
            name='Daily Feedback Check and Auto-Export',
            replace_existing=True
        )

        # Also run on startup (after 30 seconds to let app initialize)
        self.scheduler.add_job(
            self.check_and_export,
            'date',
            run_date=datetime.now(),
            id='startup_check',
            name='Startup Feedback Check'
        )

        self.scheduler.start()
        logger.info("✅ Feedback monitor started - checking daily at 9 AM")
        logger.info(f"   Threshold: {self.min_corrections} corrections")
        logger.info(f"   Export directory: {os.getenv('EXPORT_DIR', '/tmp/feedback_exports')}")

    def stop(self):
        """Stop the scheduler."""
        self.scheduler.shutdown()
        logger.info("Feedback monitor stopped")


# Global instance (will be initialized in app.py)
feedback_monitor = None


def init_monitor(session_factory, min_corrections: int = 50):
    """Initialize and start the feedback monitor."""
    global feedback_monitor

    if feedback_monitor is None:
        feedback_monitor = FeedbackMonitor(session_factory, min_corrections)
        feedback_monitor.start()
        return feedback_monitor

    return feedback_monitor


def shutdown_monitor():
    """Shutdown the feedback monitor."""
    global feedback_monitor

    if feedback_monitor:
        feedback_monitor.stop()
        feedback_monitor = None
