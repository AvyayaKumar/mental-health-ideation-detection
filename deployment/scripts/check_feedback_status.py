"""
Check feedback status and notify when ready to retrain.

This script checks your production database for feedback stats
and tells you when you have enough data to retrain.

Usage:
    # Quick check
    python scripts/check_feedback_status.py

    # Check with custom threshold
    python scripts/check_feedback_status.py --min-samples 100

    # Check and auto-download if ready
    python scripts/check_feedback_status.py --auto-download

    # Send email notification (requires setup)
    python scripts/check_feedback_status.py --email your@email.com

Can be run as a cron job:
    # Check daily at 9 AM
    0 9 * * * cd /path/to/deployment && python scripts/check_feedback_status.py --email you@email.com
"""

import os
import sys
import argparse
import requests
from datetime import datetime


def check_status(base_url: str, min_samples: int = 50):
    """
    Check feedback status from production API.

    Args:
        base_url: Base URL of your website (e.g., https://mentalhealthideation.com)
        min_samples: Minimum number of corrections needed for retraining

    Returns:
        Dictionary with status info
    """
    try:
        response = requests.get(f"{base_url}/api/v1/feedback/stats", timeout=10)
        response.raise_for_status()
        stats = response.json()

        total = stats['total_feedback']
        incorrect = stats['incorrect_predictions']
        accuracy = stats['accuracy_rate']

        is_ready = incorrect >= min_samples
        samples_needed = max(0, min_samples - incorrect)

        return {
            'ready': is_ready,
            'total_feedback': total,
            'incorrect_predictions': incorrect,
            'accuracy_rate': accuracy,
            'samples_needed': samples_needed,
            'min_samples': min_samples,
            'stats': stats
        }

    except Exception as e:
        print(f"❌ Error checking feedback status: {e}")
        return None


def download_feedback(base_url: str, output_file: str = "feedback_corrections.csv"):
    """Download feedback corrections CSV."""
    try:
        url = f"{base_url}/api/v1/feedback/export?format=csv&is_correct=false"
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        with open(output_file, 'wb') as f:
            f.write(response.content)

        print(f"✓ Downloaded corrections to: {output_file}")
        return True

    except Exception as e:
        print(f"❌ Error downloading feedback: {e}")
        return False


def send_email_notification(email: str, status: dict):
    """
    Send email notification (requires SMTP setup).

    For now, this just prints instructions. To enable emails:
    1. Set up SMTP credentials (Gmail, SendGrid, etc.)
    2. Implement email sending logic
    """
    print(f"\n📧 Email Notification (to: {email})")
    print("=" * 60)
    print("Note: Email notifications not yet configured.")
    print("To enable emails, configure SMTP settings in this script.")
    print("\nFor now, here's what would be sent:")
    print(f"\nSubject: {'✅ Ready to Retrain' if status['ready'] else '📊 Feedback Status Update'}")
    print(f"Message: You have {status['incorrect_predictions']} corrections.")
    if status['ready']:
        print("You can now retrain your model!")
    else:
        print(f"Need {status['samples_needed']} more corrections to retrain.")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Check feedback status and retraining readiness')
    parser.add_argument('--url', type=str, default='https://mentalhealthideation.com',
                       help='Base URL of your website')
    parser.add_argument('--min-samples', type=int, default=50,
                       help='Minimum number of corrections needed (default: 50)')
    parser.add_argument('--auto-download', action='store_true',
                       help='Automatically download corrections if ready')
    parser.add_argument('--email', type=str, default=None,
                       help='Send notification to this email')
    parser.add_argument('--quiet', action='store_true',
                       help='Only output if ready to retrain')

    args = parser.parse_args()

    if not args.quiet:
        print(f"\n{'='*60}")
        print("FEEDBACK STATUS CHECK")
        print(f"{'='*60}\n")
        print(f"Checking: {args.url}")
        print(f"Minimum corrections needed: {args.min_samples}\n")

    # Check status
    status = check_status(args.url, args.min_samples)

    if not status:
        return 1

    # Display results
    if not args.quiet:
        print(f"📊 Feedback Statistics:")
        print(f"  Total feedback: {status['total_feedback']}")
        print(f"  Incorrect predictions: {status['incorrect_predictions']}")
        print(f"  Model accuracy: {status['accuracy_rate']}%")
        print()

    if status['ready']:
        print(f"✅ READY TO RETRAIN!")
        print(f"   You have {status['incorrect_predictions']} corrections.")
        print(f"   This is enough to improve your model.\n")

        if not args.quiet:
            print(f"Next steps:")
            print(f"  1. Download corrections:")
            print(f"     curl '{args.url}/api/v1/feedback/export?format=csv&is_correct=false' > corrections.csv")
            print(f"  2. Follow the retraining guide:")
            print(f"     See deployment/RETRAINING_GUIDE.md")
            print()

        # Auto-download if requested
        if args.auto_download:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"feedback_corrections_{timestamp}.csv"
            if download_feedback(args.url, filename):
                print(f"\n✓ Corrections saved to: {filename}")
                print(f"  Ready to retrain!")

    else:
        if not args.quiet:
            print(f"📊 COLLECTING FEEDBACK")
            print(f"   Current: {status['incorrect_predictions']} corrections")
            print(f"   Needed: {status['samples_needed']} more")
            print(f"   Progress: {status['incorrect_predictions']}/{args.min_samples}")
            print()
            print(f"Keep using the website to collect more feedback.")
            print(f"Retraining works best with at least {args.min_samples} corrections.")

    # Send email if requested
    if args.email:
        send_email_notification(args.email, status)

    # Admin dashboard link
    if not args.quiet:
        print(f"\n💻 View dashboard: {args.url}/admin/feedback")
        print()

    # Exit code: 0 if ready, 1 if not ready
    return 0 if status['ready'] else 1


if __name__ == '__main__':
    sys.exit(main())
