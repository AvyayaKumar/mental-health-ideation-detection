"""
Export feedback data from production database.

This script connects to your Railway PostgreSQL database and exports
all feedback for local retraining.

Usage:
    # Export all feedback
    python scripts/export_feedback.py --output feedback_data.csv

    # Export only incorrect predictions (corrections)
    python scripts/export_feedback.py --output feedback_data.csv --corrections-only

    # Use custom database URL
    python scripts/export_feedback.py --database-url "postgresql://..." --output feedback_data.csv
"""

import os
import sys
import argparse
import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'backend'))

from models.database import Feedback


def export_feedback(database_url: str, output_path: str, corrections_only: bool = False):
    """
    Export feedback from database to CSV.

    Args:
        database_url: PostgreSQL connection string
        output_path: Output CSV file path
        corrections_only: Only export incorrect predictions (teacher corrections)
    """
    print(f"Connecting to database...")

    # Create engine
    engine = create_engine(database_url)
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        # Query feedback
        query = session.query(Feedback)

        if corrections_only:
            query = query.filter(Feedback.is_correct == False)
            print("Exporting only corrections (incorrect predictions)...")
        else:
            print("Exporting all feedback...")

        feedbacks = query.all()

        if not feedbacks:
            print("❌ No feedback data found in database!")
            return False

        print(f"✓ Found {len(feedbacks)} feedback records")

        # Convert to DataFrame
        data = []
        for fb in feedbacks:
            # Map risk levels to binary labels for retraining
            # HIGH = suicide, MEDIUM = suicide (to be conservative), LOW = non-suicide
            if fb.teacher_correction in ['HIGH', 'MEDIUM']:
                label = 'suicide'
            else:
                label = 'non-suicide'

            data.append({
                'text': fb.text,
                'class': label,
                'original_prediction': fb.original_prediction,
                'teacher_correction': fb.teacher_correction,
                'is_correct': fb.is_correct,
                'created_at': fb.created_at,
                'feedback_id': fb.id
            })

        df = pd.DataFrame(data)

        # Print statistics
        print(f"\nFeedback statistics:")
        print(f"  Total records: {len(df)}")
        print(f"  Correct predictions: {df['is_correct'].sum()}")
        print(f"  Incorrect predictions: {(~df['is_correct']).sum()}")
        print(f"\nClass distribution (after mapping to binary):")
        print(df['class'].value_counts())
        print(f"\nOriginal risk level distribution:")
        print(df['teacher_correction'].value_counts())

        # Save to CSV
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save with metadata
        df.to_csv(output_path, index=False)
        print(f"\n✓ Feedback data exported to: {output_path}")

        # Also save simplified version for direct retraining (just text + class)
        simple_path = output_path.parent / f"{output_path.stem}_simple{output_path.suffix}"
        df[['text', 'class']].to_csv(simple_path, index=False)
        print(f"✓ Simplified version saved to: {simple_path}")

        return True

    except Exception as e:
        print(f"❌ Error exporting feedback: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        session.close()


def main():
    parser = argparse.ArgumentParser(description='Export feedback from production database')
    parser.add_argument('--database-url', type=str, default=None,
                       help='PostgreSQL connection string (or set DATABASE_URL env var)')
    parser.add_argument('--output', type=str, default='feedback_export.csv',
                       help='Output CSV file path')
    parser.add_argument('--corrections-only', action='store_true',
                       help='Only export incorrect predictions (corrections)')

    args = parser.parse_args()

    # Get database URL
    database_url = args.database_url or os.getenv('DATABASE_URL')

    if not database_url:
        print("❌ Error: DATABASE_URL not provided!")
        print("\nProvide database URL via:")
        print("  1. --database-url argument")
        print("  2. DATABASE_URL environment variable")
        print("\nGet your DATABASE_URL from Railway dashboard:")
        print("  Dashboard → PostgreSQL → Variables → DATABASE_URL")
        return 1

    # PostgreSQL fix for SQLAlchemy 1.4+
    if database_url.startswith('postgres://'):
        database_url = database_url.replace('postgres://', 'postgresql://', 1)

    print(f"\n{'='*60}")
    print("FEEDBACK DATA EXPORT")
    print(f"{'='*60}\n")

    success = export_feedback(database_url, args.output, args.corrections_only)

    if success:
        print(f"\n{'='*60}")
        print("✓ EXPORT COMPLETE!")
        print(f"{'='*60}")
        print(f"\nNext steps:")
        print(f"  1. Review the exported data: {args.output}")
        print(f"  2. Run retraining script:")
        print(f"     python scripts/retrain_model.py --feedback-data {args.output}")
        return 0
    else:
        return 1


if __name__ == '__main__':
    sys.exit(main())
