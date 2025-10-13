"""
Email notification system for feedback and crash reports.

This module handles sending email notifications for:
- New teacher feedback submissions
- Crash reports
- System alerts
"""

import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional, Dict, Any
from datetime import datetime


class EmailNotifier:
    """Email notification service."""

    def __init__(
        self,
        smtp_host: Optional[str] = None,
        smtp_port: Optional[int] = None,
        smtp_user: Optional[str] = None,
        smtp_password: Optional[str] = None,
        admin_email: Optional[str] = None
    ):
        """
        Initialize email notifier.

        Args:
            smtp_host: SMTP server hostname (default: from SMTP_HOST env var)
            smtp_port: SMTP server port (default: from SMTP_PORT env var or 587)
            smtp_user: SMTP username (default: from SMTP_USER env var)
            smtp_password: SMTP password (default: from SMTP_PASSWORD env var)
            admin_email: Admin email to receive notifications (default: avyaya.kumar@gmail.com)
        """
        self.smtp_host = smtp_host or os.environ.get('SMTP_HOST', 'smtp.gmail.com')
        self.smtp_port = smtp_port or int(os.environ.get('SMTP_PORT', '587'))
        self.smtp_user = smtp_user or os.environ.get('SMTP_USER', '')
        self.smtp_password = smtp_password or os.environ.get('SMTP_PASSWORD', '')
        self.admin_email = admin_email or os.environ.get('ADMIN_EMAIL', 'avyaya.kumar@gmail.com')

        # Enable/disable notifications via env var
        self.enabled = os.environ.get('EMAIL_NOTIFICATIONS_ENABLED', 'false').lower() == 'true'

    def send_email(self, to_email: str, subject: str, body_html: str) -> bool:
        """
        Send an email.

        Args:
            to_email: Recipient email address
            subject: Email subject
            body_html: HTML email body

        Returns:
            True if sent successfully, False otherwise
        """
        if not self.enabled:
            print(f"[Email] Notifications disabled. Would send to {to_email}: {subject}")
            return False

        if not self.smtp_user or not self.smtp_password:
            print("[Email] SMTP credentials not configured. Set SMTP_USER and SMTP_PASSWORD env vars.")
            return False

        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.smtp_user
            msg['To'] = to_email

            # Attach HTML body
            html_part = MIMEText(body_html, 'html')
            msg.attach(html_part)

            # Send email
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)

            print(f"[Email] ✓ Sent to {to_email}: {subject}")
            return True

        except Exception as e:
            print(f"[Email] ✗ Failed to send to {to_email}: {e}")
            return False

    def notify_feedback_submitted(self, feedback_data: Dict[str, Any]) -> bool:
        """
        Send notification when teacher feedback is submitted.

        Args:
            feedback_data: Dictionary with feedback details

        Returns:
            True if sent successfully
        """
        subject = f"New Teacher Feedback - {feedback_data.get('feedback_type', 'Unknown').title()}"

        body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; padding: 20px;">
            <h2 style="color: #667eea;">New Feedback Received</h2>

            <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 20px 0;">
                <p><strong>Feedback Type:</strong> {feedback_data.get('feedback_type', 'Unknown')}</p>
                <p><strong>Prediction ID:</strong> {feedback_data.get('prediction_id', 'N/A')}</p>
                <p><strong>Timestamp:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>

            <h3>Original Prediction</h3>
            <p><strong>Predicted Class:</strong> {feedback_data.get('predicted_label', 'Unknown')}</p>
            <p><strong>Confidence:</strong> {feedback_data.get('confidence', 0):.2%}</p>

            <h3>Teacher Correction</h3>
            <p><strong>Correct Class:</strong> {feedback_data.get('correct_label', 'Unknown')}</p>

            {f"<p><strong>Teacher Notes:</strong> {feedback_data.get('teacher_notes', '')}</p>" if feedback_data.get('teacher_notes') else ''}

            {f"<p><strong>Teacher Email:</strong> {feedback_data.get('teacher_email', '')}</p>" if feedback_data.get('teacher_email') else ''}

            <div style="margin-top: 30px; padding: 15px; background: #e7f3ff; border-left: 4px solid #667eea;">
                <p style="margin: 0;"><strong>Action Required:</strong> Review this feedback in the admin dashboard at <a href="http://localhost:8080/admin">Admin Dashboard</a></p>
            </div>
        </body>
        </html>
        """

        return self.send_email(self.admin_email, subject, body)

    def notify_crash_report(self, error_details: Dict[str, Any]) -> bool:
        """
        Send notification when a crash occurs.

        Args:
            error_details: Dictionary with error details

        Returns:
            True if sent successfully
        """
        subject = f"🚨 Crash Report - {error_details.get('message', 'Unknown Error')}"

        body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; padding: 20px;">
            <h2 style="color: #dc3545;">Crash Report</h2>

            <div style="background: #f8d7da; padding: 15px; border-radius: 8px; border-left: 4px solid #dc3545; margin: 20px 0;">
                <p><strong>⚠️ A crash has been reported in the application</strong></p>
                <p><strong>Timestamp:</strong> {error_details.get('timestamp', datetime.now().isoformat())}</p>
            </div>

            <h3>Error Details</h3>
            <div style="background: #f8f9fa; padding: 15px; border-radius: 8px;">
                <p><strong>Message:</strong> {error_details.get('message', 'Unknown')}</p>
                <p><strong>User Agent:</strong> {error_details.get('userAgent', 'Unknown')}</p>
            </div>

            {f'''
            <h3>Stack Trace</h3>
            <pre style="background: #f8f9fa; padding: 15px; border-radius: 8px; overflow-x: auto;">
{error_details.get('stack', 'No stack trace available')}
            </pre>
            ''' if error_details.get('stack') else ''}

            <div style="margin-top: 30px; padding: 15px; background: #fff3cd; border-left: 4px solid #ffc107;">
                <p style="margin: 0;"><strong>Action Required:</strong> Investigate and fix this issue immediately.</p>
            </div>
        </body>
        </html>
        """

        return self.send_email(self.admin_email, subject, body)

    def notify_contact_submission(self, submission_data: Dict[str, Any]) -> bool:
        """
        Send notification when contact form is submitted.

        Args:
            submission_data: Dictionary with submission details

        Returns:
            True if sent successfully
        """
        subject = f"Contact Form - {submission_data.get('subject', 'No Subject')}"

        body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; padding: 20px;">
            <h2 style="color: #667eea;">New Contact Form Submission</h2>

            <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 20px 0;">
                <p><strong>Type:</strong> {submission_data.get('submission_type', 'Unknown')}</p>
                <p><strong>From:</strong> {submission_data.get('name', 'Anonymous')} ({submission_data.get('email', 'No email')})</p>
                <p><strong>Timestamp:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>

            <h3>Subject</h3>
            <p>{submission_data.get('subject', 'No subject')}</p>

            <h3>Message</h3>
            <div style="background: #f8f9fa; padding: 15px; border-radius: 8px;">
                {submission_data.get('message', 'No message').replace('\n', '<br>')}
            </div>

            <div style="margin-top: 30px; padding: 15px; background: #e7f3ff; border-left: 4px solid #667eea;">
                <p style="margin: 0;"><strong>Reply to:</strong> {submission_data.get('email', 'No email provided')}</p>
            </div>
        </body>
        </html>
        """

        return self.send_email(self.admin_email, subject, body)

    def notify_retraining_complete(self, retraining_data: Dict[str, Any]) -> bool:
        """
        Send notification when model retraining is complete.

        Args:
            retraining_data: Dictionary with retraining details

        Returns:
            True if sent successfully
        """
        subject = "✓ Model Retraining Complete"

        body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; padding: 20px;">
            <h2 style="color: #28a745;">Model Retraining Complete</h2>

            <div style="background: #d4edda; padding: 15px; border-radius: 8px; border-left: 4px solid #28a745; margin: 20px 0;">
                <p><strong>✓ The model has been successfully retrained with teacher feedback</strong></p>
                <p><strong>Timestamp:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>

            <h3>Training Summary</h3>
            <div style="background: #f8f9fa; padding: 15px; border-radius: 8px;">
                <p><strong>Feedback Samples Used:</strong> {retraining_data.get('num_samples', 'Unknown')}</p>
                <p><strong>Model Name:</strong> {retraining_data.get('model_name', 'Unknown')}</p>
                <p><strong>Output Directory:</strong> {retraining_data.get('output_dir', 'Unknown')}</p>
            </div>

            {f'''
            <h3>Performance Metrics</h3>
            <div style="background: #f8f9fa; padding: 15px; border-radius: 8px;">
                <pre>{retraining_data.get('metrics', 'No metrics available')}</pre>
            </div>
            ''' if retraining_data.get('metrics') else ''}

            <div style="margin-top: 30px; padding: 15px; background: #e7f3ff; border-left: 4px solid #667eea;">
                <p style="margin: 0;"><strong>Next Steps:</strong> Update MODEL_PATH environment variable to use the new model.</p>
            </div>
        </body>
        </html>
        """

        return self.send_email(self.admin_email, subject, body)


# Singleton instance
_notifier = None


def get_notifier() -> EmailNotifier:
    """Get the global email notifier instance."""
    global _notifier
    if _notifier is None:
        _notifier = EmailNotifier()
    return _notifier
