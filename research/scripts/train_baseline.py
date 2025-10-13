"""Train TF-IDF + Logistic Regression baseline for suicide ideation detection."""

import pandas as pd
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import sys
import os
from datetime import datetime
import time

# Add parent directory to path to import from src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.metrics import calculate_metrics_from_predictions, print_confusion_matrix

# Try to import wandb, but continue if not available
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  wandb not installed, skipping experiment tracking")


def main():
    """Train and evaluate TF-IDF + Logistic Regression baseline."""

    start_time = time.time()

    print("=" * 70)
    print("TF-IDF + LOGISTIC REGRESSION BASELINE")
    print("=" * 70)

    # Initialize wandb if available
    if WANDB_AVAILABLE:
        wandb.init(
            project="suicide-ideation-detection",
            name=f"baseline-tfidf-logreg-{datetime.now().strftime('%Y%m%d-%H%M')}",
            config={
                "model_type": "TF-IDF + Logistic Regression",
                "vectorizer": "TfidfVectorizer",
                "max_features": 10000,
                "ngram_range": "(1, 2)",
                "classifier": "LogisticRegression",
                "class_weight": "balanced",
                "seed": 42,
            }
        )
        print("\n✓ Wandb initialized")

    # Paths (adjust these for Colab)
    data_path = 'Suicide_Detection_Full.csv'  # Change for Colab
    split_indices_path = 'split_indices.json'  # Change for Colab

    print(f"\nLoading data from: {data_path}")

    # Load data with robust parsing
    try:
        df = pd.read_csv(data_path, dtype={'text': str, 'class': str})
    except:
        print("Standard parsing failed, using robust parser...")
        df = pd.read_csv(
            data_path,
            dtype={'text': str, 'class': str},
            on_bad_lines='skip',
            engine='python'
        )

    df = df.dropna()
    print(f"Loaded: {len(df):,} samples")

    # Map labels
    class_mapping = {'non-suicide': 0, 'suicide': 1}
    df['label'] = df['class'].map(class_mapping)
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
    df = df.reset_index(drop=True)

    # Load split indices
    print(f"\nLoading splits from: {split_indices_path}")
    with open(split_indices_path, 'r') as f:
        splits = json.load(f)

    # Create splits
    train_df = df.loc[splits['train_indices']]
    val_df = df.loc[splits['val_indices']]
    test_df = df.loc[splits['test_indices']]

    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_df):,}")
    print(f"  Val:   {len(val_df):,}")
    print(f"  Test:  {len(test_df):,}")

    # Check class distribution in train set
    train_dist = train_df['label'].value_counts()
    print(f"\nTrain set class distribution:")
    print(f"  Class 0 (non-suicide): {train_dist[0]:,}")
    print(f"  Class 1 (suicide):     {train_dist[1]:,}")

    # TF-IDF vectorization
    print("\n" + "=" * 70)
    print("VECTORIZING TEXT (TF-IDF)")
    print("=" * 70)
    print("Max features: 10,000")
    print("N-gram range: (1, 2) - unigrams and bigrams")

    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        stop_words='english'
    )

    print("\nFitting vectorizer on training data...")
    X_train = vectorizer.fit_transform(train_df['text'])
    print(f"✓ Training set vectorized: {X_train.shape}")

    print("Transforming validation set...")
    X_val = vectorizer.transform(val_df['text'])
    print(f"✓ Validation set vectorized: {X_val.shape}")

    print("Transforming test set...")
    X_test = vectorizer.transform(test_df['text'])
    print(f"✓ Test set vectorized: {X_test.shape}")

    y_train = train_df['label'].values
    y_val = val_df['label'].values
    y_test = test_df['label'].values

    # Train logistic regression
    print("\n" + "=" * 70)
    print("TRAINING LOGISTIC REGRESSION")
    print("=" * 70)
    print("Class weight: balanced (to handle any minor imbalance)")
    print("Max iterations: 1000")
    print("Solver: liblinear")

    clf = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight='balanced',
        solver='liblinear',
        verbose=1
    )

    print("\nTraining...")
    clf.fit(X_train, y_train)
    print("✓ Training complete!")

    # Evaluate on validation set
    print("\n" + "=" * 70)
    print("VALIDATION SET EVALUATION")
    print("=" * 70)

    val_preds = clf.predict(X_val)
    val_metrics = calculate_metrics_from_predictions(y_val, val_preds)

    print(f"\nValidation Metrics:")
    print(f"  Accuracy:  {val_metrics['accuracy']:.4f}")
    print(f"  Precision: {val_metrics['precision']:.4f}")
    print(f"  Recall:    {val_metrics['recall']:.4f}")
    print(f"  F1 Score:  {val_metrics['f1']:.4f}")
    print(f"  FNR:       {val_metrics['fnr']:.4f} (False Negative Rate)")

    print(f"\nConfusion Matrix:")
    print_confusion_matrix(y_val, val_preds)

    if WANDB_AVAILABLE:
        wandb.log({
            "val_accuracy": val_metrics['accuracy'],
            "val_precision": val_metrics['precision'],
            "val_recall": val_metrics['recall'],
            "val_f1": val_metrics['f1'],
            "val_fnr": val_metrics['fnr'],
        })

    # Evaluate on test set
    print("\n" + "=" * 70)
    print("TEST SET EVALUATION")
    print("=" * 70)

    test_preds = clf.predict(X_test)
    test_metrics = calculate_metrics_from_predictions(y_test, test_preds)

    print(f"\nTest Metrics:")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1 Score:  {test_metrics['f1']:.4f}")
    print(f"  FNR:       {test_metrics['fnr']:.4f} (False Negative Rate)")

    print(f"\nDetailed Classification Report:")
    print(classification_report(y_test, test_preds, target_names=['non-suicide', 'suicide'], digits=4))

    print(f"\nConfusion Matrix:")
    print_confusion_matrix(y_test, test_preds)

    if WANDB_AVAILABLE:
        wandb.log({
            "test_accuracy": test_metrics['accuracy'],
            "test_precision": test_metrics['precision'],
            "test_recall": test_metrics['recall'],
            "test_f1": test_metrics['f1'],
            "test_fnr": test_metrics['fnr'],
        })

    # Save model
    print("\n" + "=" * 70)
    print("SAVING MODEL")
    print("=" * 70)

    os.makedirs('results/baselines', exist_ok=True)
    model_path = 'results/baselines/tfidf_logreg.pkl'

    with open(model_path, 'wb') as f:
        pickle.dump({
            'vectorizer': vectorizer,
            'classifier': clf,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics
        }, f)

    print(f"✓ Model saved to: {model_path}")

    # Save results to JSON
    results_path = 'results/baselines/baseline_results.json'
    results = {
        'model_type': 'TF-IDF + Logistic Regression',
        'timestamp': datetime.now().isoformat(),
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'config': {
            'max_features': 10000,
            'ngram_range': '(1, 2)',
            'class_weight': 'balanced',
            'solver': 'liblinear'
        }
    }

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✓ Results saved to: {results_path}")

    # Calculate training time
    end_time = time.time()
    training_time = end_time - start_time
    hours = int(training_time // 3600)
    minutes = int((training_time % 3600) // 60)
    seconds = int(training_time % 60)

    # Final summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Training time: {hours}h {minutes}m {seconds}s")
    print(f"\nBaseline Performance (Test Set):")
    print(f"  F1 Score: {test_metrics['f1']:.4f}")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  False Negative Rate: {test_metrics['fnr']:.4f}")
    print("\n💡 This is the performance floor - transformer models should beat this!")
    print("=" * 70 + "\n")

    if WANDB_AVAILABLE:
        wandb.finish()

    return test_metrics


if __name__ == "__main__":
    main()
