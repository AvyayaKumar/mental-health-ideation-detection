"""Dataset class with tokenization and caching for suicide ideation detection."""

import pandas as pd
import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from pathlib import Path
import pickle
import numpy as np


class SuicideDetectionDataset(Dataset):
    """
    Dataset for suicide ideation detection with automatic tokenization and caching.

    Features:
    - Loads data from CSV using saved split indices
    - Tokenizes text using HuggingFace tokenizers
    - Caches tokenized data to disk for faster loading
    - Handles class mapping (suicide/non-suicide -> 0/1)
    """

    def __init__(
        self,
        data_path,
        split_indices,
        tokenizer_name,
        max_length=256,
        cache_dir=None,
        split_name="train",
    ):
        """
        Initialize dataset.

        Args:
            data_path: Path to CSV file (e.g., 'Suicide_Detection_Full.csv')
            split_indices: List of indices for this split
            tokenizer_name: Name of HuggingFace tokenizer (e.g., 'bert-base-uncased')
            max_length: Maximum sequence length for tokenization
            cache_dir: Directory to cache tokenized data (optional but recommended)
            split_name: Name of split for logging ('train', 'val', or 'test')
        """
        self.max_length = max_length
        self.split_name = split_name

        print(f"\n{'='*60}")
        print(f"Loading {split_name.upper()} dataset")
        print(f"{'='*60}")

        # Load data with robust parsing
        print(f"Reading CSV from: {data_path}")
        try:
            df = pd.read_csv(data_path, dtype={'text': str, 'class': str})
        except:
            # Fallback to robust parsing
            print("Standard parsing failed, using robust parser...")
            df = pd.read_csv(
                data_path,
                dtype={'text': str, 'class': str},
                on_bad_lines='skip',
                engine='python'
            )

        # Clean data
        df = df.dropna()

        # Map labels to integers
        class_mapping = {'non-suicide': 0, 'suicide': 1}
        df['label'] = df['class'].map(class_mapping)
        df = df.dropna(subset=['label'])
        df['label'] = df['label'].astype(int)
        df = df.reset_index(drop=True)

        # Get split
        self.df = df.loc[split_indices].reset_index(drop=True)
        print(f"Loaded {len(self.df):,} samples")

        # Check class distribution
        class_dist = self.df['label'].value_counts()
        print(f"\nClass distribution:")
        print(f"  Non-suicide (0): {class_dist.get(0, 0):,}")
        print(f"  Suicide (1):     {class_dist.get(1, 0):,}")

        # Initialize tokenizer
        print(f"\nInitializing tokenizer: {tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # Check cache
        cache_file = None
        if cache_dir:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            # Create unique cache filename based on model, length, and split size
            cache_filename = f"{tokenizer_name.replace('/', '_')}_{max_length}_{split_name}_{len(split_indices)}.pkl"
            cache_file = Path(cache_dir) / cache_filename

            if cache_file.exists():
                print(f"\n✓ Loading from cache: {cache_file.name}")
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.encodings = cache_data['encodings']
                    self.labels = cache_data['labels']
                print(f"✓ Cached data loaded successfully!")
                return

        # Tokenize (if not cached)
        print(f"\nTokenizing {len(self.df):,} samples...")
        print(f"Max length: {max_length}")

        texts = self.df['text'].tolist()
        self.labels = self.df['label'].tolist()

        # Tokenize in batches to show progress
        self.encodings = self._tokenize_in_batches(texts, batch_size=5000)

        # Save to cache
        if cache_file:
            print(f"\n✓ Saving to cache: {cache_file.name}")
            cache_data = {
                'encodings': self.encodings,
                'labels': self.labels,
                'tokenizer_name': tokenizer_name,
                'max_length': max_length,
                'split_name': split_name,
                'num_samples': len(self.labels)
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"✓ Cache saved successfully!")

        print(f"{'='*60}\n")

    def _tokenize_in_batches(self, texts, batch_size=5000):
        """Tokenize texts in batches to show progress and manage memory."""
        all_encodings = None
        total_batches = (len(texts) + batch_size - 1) // batch_size

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]

            if i % (batch_size * 2) == 0 or i == 0:  # Print every 2 batches
                print(f"  Tokenizing batch {i//batch_size + 1}/{total_batches} ({i:,}/{len(texts):,} samples)...")

            batch_encodings = self.tokenizer(
                batch_texts,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )

            if all_encodings is None:
                all_encodings = batch_encodings
            else:
                for key in all_encodings:
                    all_encodings[key] = torch.cat([all_encodings[key], batch_encodings[key]])

        print(f"✓ Tokenization complete!")
        return all_encodings

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """Get a single sample."""
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def load_datasets(data_path, split_indices_path, tokenizer_name, max_length=256, cache_dir=None):
    """
    Load train, val, and test datasets.

    Args:
        data_path: Path to CSV file
        split_indices_path: Path to JSON file with split indices
        tokenizer_name: Name of HuggingFace tokenizer
        max_length: Maximum sequence length
        cache_dir: Directory for caching tokenized data

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    # Load split indices
    print(f"Loading split indices from: {split_indices_path}")
    with open(split_indices_path, 'r') as f:
        splits = json.load(f)

    print(f"\nSplit information:")
    print(f"  Total samples: {splits['total_samples']:,}")
    print(f"  Train: {splits['train_size']:,} ({splits['train_size']/splits['total_samples']*100:.1f}%)")
    print(f"  Val:   {splits['val_size']:,} ({splits['val_size']/splits['total_samples']*100:.1f}%)")
    print(f"  Test:  {splits['test_size']:,} ({splits['test_size']/splits['total_samples']*100:.1f}%)")
    print(f"  Random seed: {splits['random_seed']}")

    # Create datasets
    train_dataset = SuicideDetectionDataset(
        data_path=data_path,
        split_indices=splits['train_indices'],
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        cache_dir=cache_dir,
        split_name='train'
    )

    val_dataset = SuicideDetectionDataset(
        data_path=data_path,
        split_indices=splits['val_indices'],
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        cache_dir=cache_dir,
        split_name='val'
    )

    test_dataset = SuicideDetectionDataset(
        data_path=data_path,
        split_indices=splits['test_indices'],
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        cache_dir=cache_dir,
        split_name='test'
    )

    return train_dataset, val_dataset, test_dataset
