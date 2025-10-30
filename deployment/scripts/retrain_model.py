"""
Retrain RoBERTa model with teacher feedback data.

This script combines the original training data with teacher feedback
and retrains the model to improve accuracy on real-world edge cases.

Usage:
    # Retrain with feedback data
    python scripts/retrain_model.py --feedback-data feedback_export_simple.csv

    # Custom settings
    python scripts/retrain_model.py \
        --feedback-data feedback_export_simple.csv \
        --original-data ../research/data/raw/Suicide_Detection_Full.csv \
        --output-dir retrained_roberta_v2 \
        --oversample 5 \
        --epochs 3

Requirements:
    - Original training data (Suicide_Detection_Full.csv)
    - Feedback data exported from production
    - Sufficient GPU/CPU for training (can take 1-2 hours)
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


def load_and_combine_data(original_path: str, feedback_path: str, oversample: int = 5):
    """
    Load original training data and feedback, then combine them.

    Args:
        original_path: Path to original training CSV
        feedback_path: Path to feedback CSV (text, class columns)
        oversample: Number of times to duplicate feedback samples (increases their weight)

    Returns:
        Combined DataFrame
    """
    print(f"Loading original training data from: {original_path}")
    original_df = pd.read_csv(original_path)

    # Standardize column names
    if 'text' in original_df.columns and 'class' in original_df.columns:
        original_df = original_df[['text', 'class']]
    else:
        raise ValueError("Original data must have 'text' and 'class' columns")

    print(f"✓ Loaded {len(original_df)} original samples")
    print(f"  Original distribution: {original_df['class'].value_counts().to_dict()}")

    # Load feedback
    print(f"\nLoading feedback data from: {feedback_path}")
    feedback_df = pd.read_csv(feedback_path)

    if 'text' not in feedback_df.columns or 'class' not in feedback_df.columns:
        raise ValueError("Feedback data must have 'text' and 'class' columns")

    feedback_df = feedback_df[['text', 'class']]
    print(f"✓ Loaded {len(feedback_df)} feedback samples")
    print(f"  Feedback distribution: {feedback_df['class'].value_counts().to_dict()}")

    # Oversample feedback to give it more weight in training
    if oversample > 1:
        print(f"\nOversampling feedback {oversample}x to increase training weight...")
        feedback_df = pd.concat([feedback_df] * oversample, ignore_index=True)
        print(f"✓ Feedback after oversampling: {len(feedback_df)} samples")

    # Combine datasets
    combined_df = pd.concat([original_df, feedback_df], ignore_index=True)
    combined_df = combined_df.dropna()  # Remove any NaN values

    print(f"\n✓ Combined dataset: {len(combined_df)} total samples")
    print(f"  Combined distribution: {combined_df['class'].value_counts().to_dict()}")

    return combined_df


def prepare_dataset(df: pd.DataFrame, tokenizer, max_length: int = 256):
    """
    Tokenize and prepare dataset for training.

    Args:
        df: DataFrame with 'text' and 'class' columns
        tokenizer: Hugging Face tokenizer
        max_length: Maximum sequence length

    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    print(f"\nPreparing dataset...")

    # Map labels to integers
    label_map = {'non-suicide': 0, 'suicide': 1}
    df['label'] = df['class'].map(label_map)

    # Check for unmapped labels
    if df['label'].isna().any():
        unmapped = df[df['label'].isna()]['class'].unique()
        raise ValueError(f"Unmapped labels found: {unmapped}")

    # Split data (90% train, 10% eval)
    train_df, eval_df = train_test_split(
        df[['text', 'label']],
        test_size=0.1,
        random_state=42,
        stratify=df['label']
    )

    print(f"✓ Train: {len(train_df)} samples")
    print(f"✓ Eval: {len(eval_df)} samples")

    # Convert to Hugging Face Dataset
    train_dataset = Dataset.from_pandas(train_df)
    eval_dataset = Dataset.from_pandas(eval_df)

    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding=False,
            truncation=True,
            max_length=max_length
        )

    print(f"Tokenizing...")
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    eval_dataset = eval_dataset.map(tokenize_function, batched=True)

    print(f"✓ Tokenization complete")

    return train_dataset, eval_dataset


def compute_metrics_fn(eval_pred):
    """Compute evaluation metrics."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary', pos_label=1
    )

    # Confusion matrix for FNR
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'fnr': fnr
    }


def train_model(
    train_dataset,
    eval_dataset,
    model_name: str = "roberta-base",
    output_dir: str = "retrained_model",
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5
):
    """
    Train the model.

    Args:
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        model_name: Base model to fine-tune
        output_dir: Directory to save the model
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
    """
    print(f"\n{'='*60}")
    print("STARTING MODEL TRAINING")
    print(f"{'='*60}\n")

    # Load model
    print(f"Loading base model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        problem_type="single_label_classification"
    )

    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        push_to_hub=False,
        logging_dir=f"{output_dir}/logs",
        logging_steps=100,
        save_total_limit=2,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
    )

    # Train
    print(f"\nTraining for {epochs} epochs...")
    print(f"This may take 1-2 hours depending on hardware...\n")

    train_result = trainer.train()

    # Evaluate
    print(f"\nEvaluating final model...")
    eval_results = trainer.evaluate()

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE!")
    print(f"{'='*60}\n")

    print("Final metrics:")
    for key, value in eval_results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    # Save final model
    final_model_dir = f"{output_dir}/final_model"
    print(f"\nSaving model to: {final_model_dir}")
    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)

    print(f"✓ Model saved successfully!")

    return eval_results


def main():
    parser = argparse.ArgumentParser(description='Retrain model with feedback data')
    parser.add_argument('--feedback-data', type=str, required=True,
                       help='Path to feedback CSV (text, class columns)')
    parser.add_argument('--original-data', type=str,
                       default='../research/data/raw/Suicide_Detection_Full.csv',
                       help='Path to original training data')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for retrained model')
    parser.add_argument('--oversample', type=int, default=5,
                       help='Oversample feedback N times (default: 5)')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of training epochs (default: 3)')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Training batch size (default: 16)')
    parser.add_argument('--learning-rate', type=float, default=2e-5,
                       help='Learning rate (default: 2e-5)')
    parser.add_argument('--model-name', type=str, default='roberta-base',
                       help='Base model to fine-tune (default: roberta-base)')

    args = parser.parse_args()

    # Create output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"retrained_roberta_{timestamp}"

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print("MODEL RETRAINING WITH FEEDBACK")
    print(f"{'='*60}\n")

    # Check CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cpu":
        print("⚠️  WARNING: Training on CPU will be very slow!")
        print("    Consider using Google Colab with GPU for faster training.\n")

    # Load and combine data
    combined_df = load_and_combine_data(
        args.original_data,
        args.feedback_data,
        args.oversample
    )

    # Prepare dataset
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_dataset, eval_dataset = prepare_dataset(combined_df, tokenizer)

    # Train model
    results = train_model(
        train_dataset,
        eval_dataset,
        model_name=args.model_name,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )

    # Save training info
    info = {
        'timestamp': datetime.now().isoformat(),
        'original_data': args.original_data,
        'feedback_data': args.feedback_data,
        'oversample': args.oversample,
        'total_samples': len(combined_df),
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'model_name': args.model_name,
        'final_metrics': {k: float(v) if isinstance(v, (np.floating, float)) else v
                         for k, v in results.items()}
    }

    import json
    info_path = f"{args.output_dir}/training_info.json"
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)

    print(f"\n{'='*60}")
    print("✓ RETRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"\nModel saved to: {args.output_dir}/final_model")
    print(f"Training info: {info_path}")
    print(f"\nTo deploy this model:")
    print(f"  1. Zip the model: zip -r retrained_model.zip {args.output_dir}/final_model")
    print(f"  2. Upload to Google Drive")
    print(f"  3. Update GDRIVE_FILE_ID in Railway")
    print(f"  4. Redeploy your service")

    return 0


if __name__ == '__main__':
    sys.exit(main())
