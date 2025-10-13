"""Evaluation metrics for suicide ideation detection."""

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
import numpy as np


def compute_metrics(pred):
    """
    Compute metrics for Hugging Face Trainer.

    This function is called by the Trainer during evaluation.

    Args:
        pred: Prediction object with:
            - predictions: model logits (before argmax)
            - label_ids: true labels

    Returns:
        dict: Dictionary of metric names and values
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    # Calculate standard metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary', zero_division=0
    )
    acc = accuracy_score(labels, preds)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()

    # False Negative Rate (CRITICAL for suicide detection)
    # FNR = FN / (FN + TP) = missed positives / total positives
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    # False Positive Rate
    # FPR = FP / (FP + TN) = false alarms / total negatives
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    return {
        'accuracy': float(acc),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'fnr': float(fnr),  # Most critical metric
        'fpr': float(fpr),
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),  # Most critical errors
    }


def get_classification_report(y_true, y_pred, target_names=None):
    """
    Get detailed classification report.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        target_names: Class names for report

    Returns:
        str: Formatted classification report
    """
    if target_names is None:
        target_names = ['non-suicide', 'suicide']

    return classification_report(
        y_true, y_pred,
        target_names=target_names,
        digits=4
    )


def print_confusion_matrix(y_true, y_pred, class_names=None):
    """
    Print confusion matrix in readable format.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names for classes
    """
    if class_names is None:
        class_names = ['non-suicide', 'suicide']

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print("=" * 50)
    print("CONFUSION MATRIX")
    print("=" * 50)
    print(f"\n{'':>15} {'Predicted Neg':>15} {'Predicted Pos':>15}")
    print(f"{'Actual Neg':>15} {tn:>15} {fp:>15}")
    print(f"{'Actual Pos':>15} {fn:>15} {tp:>15}")

    print(f"\n{'':<20} {'Count':>10} {'Meaning':>30}")
    print("-" * 60)
    print(f"{'True Negatives (TN)':<20} {tn:>10} {'Correctly identified non-suicide':>30}")
    print(f"{'False Positives (FP)':<20} {fp:>10} {'Non-suicide flagged as suicide':>30}")
    print(f"{'False Negatives (FN)':<20} {fn:>10} {'⚠️  MISSED suicide cases':>30}")
    print(f"{'True Positives (TP)':<20} {tp:>10} {'Correctly identified suicide':>30}")

    # Calculate rates
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    print(f"\n{'Metric':<30} {'Value':>10} {'Interpretation':>30}")
    print("-" * 70)
    print(f"{'False Negative Rate (FNR)':<30} {fnr:>10.2%} {'% of suicide cases MISSED':>30}")
    print(f"{'False Positive Rate (FPR)':<30} {fpr:>10.2%} {'% of non-suicide flagged':>30}")
    print("=" * 50)


def calculate_metrics_from_predictions(y_true, y_pred):
    """
    Calculate all metrics from true and predicted labels.

    Args:
        y_true: True labels (array-like)
        y_pred: Predicted labels (array-like)

    Returns:
        dict: All computed metrics
    """
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Standard metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Rates
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # Same as recall
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0  # Specificity

    return {
        'accuracy': float(acc),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'fnr': float(fnr),
        'fpr': float(fpr),
        'tpr': float(tpr),
        'tnr': float(tnr),
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'total_samples': len(y_true),
        'positive_samples': int((y_true == 1).sum()),
        'negative_samples': int((y_true == 0).sum()),
    }
