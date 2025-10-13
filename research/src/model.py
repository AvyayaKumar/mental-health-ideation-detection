"""Model loading utilities for transformer models."""

from transformers import AutoModelForSequenceClassification


def load_model(model_config, num_labels=2):
    """
    Load pretrained transformer model for sequence classification.

    Args:
        model_config: Model configuration dict (from YAML)
        num_labels: Number of output classes (default: 2 for binary classification)

    Returns:
        model: Pretrained transformer model
    """
    model_name = model_config['model']['name']
    model_type = model_config['model']['type']

    print(f"\n{'='*60}")
    print(f"Loading model: {model_type.upper()}")
    print(f"{'='*60}")
    print(f"Model name: {model_name}")
    print(f"Model type: {model_type}")
    print(f"Parameters: {model_config['model'].get('parameters', 'Unknown')}")
    print(f"Num labels: {num_labels}")

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )

    print(f"✓ Model loaded successfully!")
    print(f"{'='*60}\n")

    return model
