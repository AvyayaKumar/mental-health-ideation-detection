"""
Retrain model using teacher feedback data.

This script:
1. Loads feedback corrections from the database
2. Combines with original training data
3. Retrains the model with the enhanced dataset
4. Evaluates performance and logs results
5. Saves the improved model

Usage:
    python scripts/retrain_with_feedback.py --config config/train_config.yaml --min-feedback 50
"""

import sys
import os
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.database import FeedbackDatabase
from src.dataset import SuicideDataset
from src.train import train_model
from src.metrics import compute_metrics
import yaml


def load_feedback_data(db: FeedbackDatabase, min_samples: int = 50, feedback_type: str = 'correction'):
    """
    Load feedback data from database.

    Args:
        db: Database connection
        min_samples: Minimum number of feedback samples required
        feedback_type: Type of feedback to use ('correction', 'all')

    Returns:
        DataFrame with text and labels
    """
    print(f"Loading feedback data (type: {feedback_type})...")

    if feedback_type == 'all':
        feedback_data = db.get_feedback_for_retraining()
    else:
        feedback_data = db.get_feedback_for_retraining(feedback_type=feedback_type)

    if len(feedback_data) < min_samples:
        raise ValueError(
            f"Insufficient feedback data: {len(feedback_data)} samples (minimum: {min_samples}). "
            "Need more teacher corrections before retraining."
        )

    print(f"✓ Loaded {len(feedback_data)} feedback samples")

    # Convert to DataFrame
    df = pd.DataFrame({
        'text': [item['text'] for item in feedback_data],
        'class': [item['correct_class'] for item in feedback_data],
        'feedback_id': [item['feedback_id'] for item in feedback_data]
    })

    # Print class distribution
    print(f"\nFeedback class distribution:")
    print(df['class'].value_counts())

    return df


def combine_datasets(original_data_path: str, feedback_df: pd.DataFrame, oversample_feedback: int = 3):
    """
    Combine original training data with feedback data.

    Args:
        original_data_path: Path to original training CSV
        feedback_df: DataFrame with feedback data
        oversample_feedback: How many times to duplicate feedback samples (to increase their weight)

    Returns:
        Combined DataFrame
    """
    print(f"\nLoading original training data from: {original_data_path}")

    # Load original data
    original_df = pd.read_csv(original_data_path)
    print(f"✓ Loaded {len(original_df)} original samples")

    # Oversample feedback to give it more weight
    if oversample_feedback > 1:
        print(f"Oversampling feedback data {oversample_feedback}x to increase weight...")
        feedback_df = pd.concat([feedback_df] * oversample_feedback, ignore_index=True)

    # Combine datasets
    feedback_df_minimal = feedback_df[['text', 'class']]  # Drop feedback_id for consistency
    combined_df = pd.concat([original_df, feedback_df_minimal], ignore_index=True)

    print(f"\n✓ Combined dataset: {len(combined_df)} total samples")
    print(f"  - Original: {len(original_df)}")
    print(f"  - Feedback: {len(feedback_df)} (after {oversample_feedback}x oversampling)")

    # Print class distribution
    print(f"\nCombined class distribution:")
    print(combined_df['class'].value_counts())

    return combined_df


def retrain_model(config_path: str, train_df: pd.DataFrame, output_dir: str):
    """
    Retrain model with combined dataset.

    Args:
        config_path: Path to training config YAML
        train_df: Training DataFrame
        output_dir: Directory to save retrained model

    Returns:
        Training metrics
    """
    print(f"\n{'='*60}")
    print("Starting model retraining...")
    print(f"{'='*60}")

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Override output directory
    config['output_dir'] = output_dir

    # Save combined dataset temporarily
    temp_data_path = os.path.join(output_dir, 'combined_train_data.csv')
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(temp_data_path, index=False)
    print(f"✓ Saved combined dataset to: {temp_data_path}")

    # Update config to use combined data
    config['data_path'] = temp_data_path

    # Train model
    print(f"\nTraining with config:")
    print(json.dumps(config, indent=2))

    results = train_model(config)

    print(f"\n✓ Model training complete!")
    print(f"  - Model saved to: {output_dir}")
    print(f"  - Metrics: {results}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Retrain model with teacher feedback')
    parser.add_argument('--config', type=str, default='config/train_config.yaml',
                        help='Path to training config')
    parser.add_argument('--min-feedback', type=int, default=50,
                        help='Minimum feedback samples required')
    parser.add_argument('--feedback-type', type=str, default='correction',
                        choices=['correction', 'all'],
                        help='Type of feedback to use')
    parser.add_argument('--oversample', type=int, default=3,
                        help='Oversample feedback samples N times')
    parser.add_argument('--original-data', type=str, default='data/train.csv',
                        help='Path to original training data')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for retrained model')
    parser.add_argument('--db-path', type=str, default='data/feedback.db',
                        help='Path to feedback database')

    args = parser.parse_args()

    # Create output directory with timestamp
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"results/retrained_{timestamp}"

    print(f"\n{'='*60}")
    print("MODEL RETRAINING WITH FEEDBACK")
    print(f"{'='*60}\n")

    # Initialize database
    db = FeedbackDatabase(args.db_path)

    # Load feedback data
    try:
        feedback_df = load_feedback_data(db, args.min_feedback, args.feedback_type)
    except ValueError as e:
        print(f"\n❌ Error: {e}")
        print("\nRetraining cancelled. Collect more feedback and try again.")
        return 1

    # Combine with original data
    combined_df = combine_datasets(args.original_data, feedback_df, args.oversample)

    # Retrain model
    results = retrain_model(args.config, combined_df, args.output_dir)

    # Record retraining in database
    feedback_ids = feedback_df['feedback_id'].unique().tolist()
    training_config = {
        'config_path': args.config,
        'original_data': args.original_data,
        'min_feedback': args.min_feedback,
        'feedback_type': args.feedback_type,
        'oversample': args.oversample,
        'total_samples': len(combined_df),
        'output_dir': args.output_dir
    }

    retraining_id = db.record_retraining(
        feedback_ids=feedback_ids,
        model_name='distilbert-feedback-enhanced',
        training_config=training_config,
        performance_metrics=results,
        status='completed'
    )

    print(f"\n{'='*60}")
    print(f"✓ RETRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"\nRetraining ID: {retraining_id}")
    print(f"Feedback samples used: {len(feedback_ids)}")
    print(f"Total training samples: {len(combined_df)}")
    print(f"\nNew model saved to: {args.output_dir}")
    print(f"\nTo use the new model, update your MODEL_PATH:")
    print(f"  export MODEL_PATH={args.output_dir}/final_model")
    print(f"\nOr in your .env file:")
    print(f"  MODEL_PATH={args.output_dir}/final_model")

    return 0


if __name__ == '__main__':
    exit(main())
