"""Utility functions for configuration loading, random seeds, etc."""

import yaml
import random
import numpy as np
import torch
from pathlib import Path
import os


def load_config(config_path):
    """
    Load YAML configuration file.

    Args:
        config_path: Path to YAML config file

    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def merge_configs(base_config, model_config):
    """
    Merge model-specific config with base config.
    Model config takes precedence.

    Args:
        base_config: Base configuration dict
        model_config: Model-specific configuration dict

    Returns:
        dict: Merged configuration
    """
    merged = base_config.copy()

    # Add model info
    if 'model' in model_config:
        merged['model'] = model_config['model']

    # Override training settings if specified
    if 'training' in model_config:
        if 'training' not in merged:
            merged['training'] = {}
        merged['training'].update(model_config['training'])

    return merged


def set_seed(seed):
    """
    Set random seed for reproducibility across all libraries.

    Args:
        seed: Random seed integer
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # For reproducibility on CUDA
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Set for transformers
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_device():
    """
    Get available device (GPU or CPU).

    Returns:
        torch.device: Available device
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device("cpu")
        print("⚠️  Using CPU (will be slower)")

    return device


def count_parameters(model):
    """
    Count trainable parameters in model.

    Args:
        model: PyTorch model

    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds):
    """
    Format seconds into human-readable time.

    Args:
        seconds: Time in seconds

    Returns:
        str: Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def calculate_class_weights(labels, device='cpu'):
    """
    Calculate class weights for imbalanced datasets.

    Args:
        labels: List or array of labels
        device: Device to put tensor on

    Returns:
        torch.Tensor: Class weights
    """
    unique_labels, counts = np.unique(labels, return_counts=True)
    n_samples = len(labels)
    n_classes = len(unique_labels)

    # Weight formula: n_samples / (n_classes * class_count)
    weights = n_samples / (n_classes * counts)

    # Convert to tensor
    weight_tensor = torch.FloatTensor(weights).to(device)

    print(f"Class weights calculated:")
    for label, weight, count in zip(unique_labels, weights, counts):
        print(f"  Class {label}: weight={weight:.3f} (count={count})")

    return weight_tensor
