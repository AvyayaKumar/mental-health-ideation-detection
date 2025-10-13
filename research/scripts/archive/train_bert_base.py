"""
BERT Base Training Script for Ideation Detection
================================================
Uses full BERT instead of DistilBERT for potentially higher accuracy
90/5/5 train/val/test split
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast, BertForSequenceClassification, TrainingArguments, Trainer
import torch
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import time
import gc
import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Check environment
try:
    import google.colab
    IN_COLAB = True
    print("Running in Google Colab")
except:
    IN_COLAB = False
    print("Running locally")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

class TextClassificationDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

def main():
    print("=" * 60)
    print("TRAINING BERT-BASE MODEL WITH 90/5/5 SPLIT")
    print("Expected higher accuracy than DistilBERT")
    print("=" * 60)
    
    start_time = time.time()
    
    # Find dataset
    csv_files = ['Suicide_Detection 2.csv', 'Suicide_Detection_Full.csv', 'Suicide_Detection_2.csv']
    csv_path = None
    
    for file in csv_files:
        if os.path.exists(file):
            csv_path = file
            print(f"Found dataset: {file}")
            break
    
    if csv_path is None:
        print("ERROR: Dataset not found!")
        print("Please upload 'Suicide_Detection 2.csv' to Colab")
        return
    
    # Load data
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path, dtype={'text': str, 'class': str})
    
    # Clean data
    print("Cleaning data...")
    df = df.dropna()
    
    # Map labels
    class_mapping = {'non-suicide': 0, 'suicide': 1}
    df['class'] = df['class'].map(class_mapping)
    df = df.dropna()
    df['class'] = df['class'].astype(int)
    
    print(f"Dataset loaded: {len(df)} samples")
    
    # Class distribution
    class_dist = df['class'].value_counts()
    print(f"\nClass distribution:")
    print(f"  Non-suicide (0): {class_dist.get(0, 0)} samples ({class_dist.get(0, 0)/len(df)*100:.1f}%)")
    print(f"  Suicide (1): {class_dist.get(1, 0)} samples ({class_dist.get(1, 0)/len(df)*100:.1f}%)")
    
    # 90/5/5 split
    print("\nCreating 90/5/5 train/val/test split...")
    train_df, temp_df = train_test_split(df, test_size=0.1, stratify=df['class'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['class'], random_state=42)
    
    print(f"Training samples: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Validation samples: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"Test samples: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
    
    # Clear memory
    del df
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Initialize BERT tokenizer
    print("\nInitializing BERT tokenizer...")
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    
    # Tokenize in batches
    def tokenize_in_batches(texts, batch_size=2000):
        """Tokenize in smaller batches to avoid memory issues"""
        all_encodings = None
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            if i % (batch_size * 2) == 0:
                print(f"  Tokenizing batch {i//batch_size + 1}/{total_batches}...")
            
            # BERT typically performs better with longer sequences for context
            batch_encodings = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=256,  # Longer than DistilBERT's 128 for better context
                return_tensors="pt"
            )
            
            if all_encodings is None:
                all_encodings = batch_encodings
            else:
                for key in all_encodings:
                    all_encodings[key] = torch.cat([all_encodings[key], batch_encodings[key]])
            
            # Clear cache periodically
            if torch.cuda.is_available() and i % (batch_size * 5) == 0:
                torch.cuda.empty_cache()
        
        return all_encodings
    
    print("Tokenizing training data...")
    train_encodings = tokenize_in_batches(train_df['text'].tolist())
    
    print("Tokenizing validation data...")
    val_encodings = tokenize_in_batches(val_df['text'].tolist())
    
    print("Tokenizing test data...")
    test_encodings = tokenize_in_batches(test_df['text'].tolist())
    
    # Create datasets
    train_dataset = TextClassificationDataset(train_encodings, train_df['class'].tolist())
    val_dataset = TextClassificationDataset(val_encodings, val_df['class'].tolist())
    test_dataset = TextClassificationDataset(test_encodings, test_df['class'].tolist())
    
    # Clear memory
    del train_df, val_df, test_df
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Load BERT model
    print(f"\nLoading BERT-base model to {device}...")
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=2,
        attention_probs_dropout_prob=0.2,  # Add dropout for regularization
        hidden_dropout_prob=0.2
    ).to(device)
    
    # Training arguments optimized for BERT
    if IN_COLAB and torch.cuda.is_available():
        training_args = TrainingArguments(
            output_dir="./results_bert_base",
            eval_strategy="steps",
            eval_steps=1000,
            save_strategy="steps",
            save_steps=2000,
            logging_strategy="steps",
            logging_steps=100,
            per_device_train_batch_size=8,  # Smaller batch due to larger model
            per_device_eval_batch_size=16,
            gradient_accumulation_steps=2,  # Accumulate for effective batch size of 16
            num_train_epochs=3,  # More epochs for BERT
            learning_rate=2e-5,
            warmup_ratio=0.1,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            save_total_limit=2,
            fp16=True,  # Mixed precision
            dataloader_pin_memory=True,
            report_to="none",
            seed=42,
        )
    else:
        # CPU/non-Colab settings
        training_args = TrainingArguments(
            output_dir="./results_bert_base",
            eval_strategy="steps",
            eval_steps=2000,
            save_strategy="steps",
            save_steps=2000,
            logging_strategy="steps",
            logging_steps=200,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=8,
            gradient_accumulation_steps=4,
            num_train_epochs=3,
            learning_rate=2e-5,
            warmup_ratio=0.1,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            save_total_limit=2,
            report_to="none",
            seed=42,
        )
    
    # Enhanced metrics
    def compute_metrics(pred):
        logits, labels = pred
        preds = logits.argmax(axis=1)
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
        accuracy = accuracy_score(labels, preds)
        
        # Confusion matrix for additional insights
        cm = confusion_matrix(labels, preds)
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "specificity": specificity,
        }
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )
    
    # Train
    print("\n" + "=" * 60)
    print("Starting BERT training...")
    print("Expected time: 1-2 hours on Colab GPU")
    print("=" * 60 + "\n")
    
    trainer.train()
    
    # Evaluate
    print("\nEvaluating on validation set...")
    eval_results = trainer.evaluate()
    print("\nValidation Results:")
    for metric, value in eval_results.items():
        if not metric.startswith('eval_'):
            print(f"  {metric}: {value:.4f}")
    
    print("\nEvaluating on test set...")
    test_results = trainer.evaluate(test_dataset)
    print("\nTest Results:")
    for metric, value in test_results.items():
        if not metric.startswith('eval_'):
            print(f"  {metric}: {value:.4f}")
    
    # Save model
    print("\nSaving model...")
    trainer.save_model("./results_bert_base/final_model")
    tokenizer.save_pretrained("./results_bert_base/final_model")
    
    # Training time
    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)
    
    print("\n" + "=" * 60)
    print(f"✓ BERT training completed in {hours}h {minutes}m {seconds}s")
    print(f"✓ Model saved to: ./results_bert_base/final_model")
    print("=" * 60)
    
    # Save summary
    with open("./results_bert_base/training_summary.txt", "w") as f:
        f.write(f"BERT-Base Training Summary\n")
        f.write(f"{'='*50}\n")
        f.write(f"Model: bert-base-uncased\n")
        f.write(f"Parameters: ~110M (vs DistilBERT's ~66M)\n")
        f.write(f"Max sequence length: 256 tokens\n")
        f.write(f"Train samples: {len(train_dataset)}\n")
        f.write(f"Val samples: {len(val_dataset)}\n")
        f.write(f"Test samples: {len(test_dataset)}\n")
        f.write(f"Training time: {hours}h {minutes}m {seconds}s\n")
        f.write(f"\nValidation Results:\n")
        for k, v in eval_results.items():
            f.write(f"  {k}: {v}\n")
        f.write(f"\nTest Results:\n")
        for k, v in test_results.items():
            f.write(f"  {k}: {v}\n")
    
    if IN_COLAB:
        print("\n" + "=" * 60)
        print("NEXT STEPS:")
        print("1. Run: !zip -r results_bert.zip results_bert_base")
        print("2. Run: from google.colab import files")
        print("3. Run: files.download('results_bert.zip')")
        print("=" * 60)

if __name__ == "__main__":
    main()