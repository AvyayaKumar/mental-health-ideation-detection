"""
RoBERTa Training Script for Ideation Detection - FIXED VERSION
==============================================================
RoBERTa: Robustly Optimized BERT - Better performance on classification tasks
90/5/5 train/val/test split
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, TrainingArguments, Trainer
import torch
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
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
    print("TRAINING ROBERTA MODEL WITH 90/5/5 SPLIT")
    print("RoBERTa advantages:")
    print("- Trained on more data than BERT")
    print("- Better preprocessing")
    print("- Often superior for classification tasks")
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
    
    # Check for class imbalance
    imbalance_ratio = class_dist.get(0, 0) / class_dist.get(1, 1)
    if imbalance_ratio > 2 or imbalance_ratio < 0.5:
        print(f"⚠️ Class imbalance detected (ratio: {imbalance_ratio:.2f})")
        print("  Consider using class weights for better performance")
    
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
    
    # Initialize RoBERTa tokenizer
    print("\nInitializing RoBERTa tokenizer...")
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    
    # Tokenize in batches
    def tokenize_in_batches(texts, batch_size=2000):
        """Tokenize in smaller batches to avoid memory issues"""
        all_encodings = None
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            if i % (batch_size * 2) == 0:
                print(f"  Tokenizing batch {i//batch_size + 1}/{total_batches}...")
            
            # RoBERTa handles longer sequences well
            batch_encodings = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=256,
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
    
    # Calculate class weights for imbalanced data
    train_labels = train_df['class'].values
    class_weights = len(train_labels) / (2 * np.bincount(train_labels))
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    
    # Clear memory
    del train_df, val_df, test_df
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Load RoBERTa model
    print(f"\nLoading RoBERTa model to {device}...")
    model = RobertaForSequenceClassification.from_pretrained(
        "roberta-base",
        num_labels=2,
        attention_probs_dropout_prob=0.1,
        hidden_dropout_prob=0.1,
        problem_type="single_label_classification"
    ).to(device)
    
    # Training arguments optimized for RoBERTa
    if IN_COLAB and torch.cuda.is_available():
        training_args = TrainingArguments(
            output_dir="./results_roberta",
            eval_strategy="steps",
            eval_steps=800,
            save_strategy="steps",
            save_steps=1600,
            logging_strategy="steps",
            logging_steps=80,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=16,
            gradient_accumulation_steps=2,
            num_train_epochs=3,
            learning_rate=1e-5,  # Lower LR for RoBERTa
            warmup_ratio=0.06,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            save_total_limit=2,
            fp16=True,
            dataloader_pin_memory=True,
            optim="adamw_torch",  # Better optimizer
            lr_scheduler_type="cosine",  # Cosine annealing
            report_to="none",
            seed=42,
            push_to_hub=False,
        )
    else:
        # CPU/non-Colab settings
        training_args = TrainingArguments(
            output_dir="./results_roberta",
            eval_strategy="steps",
            eval_steps=1600,
            save_strategy="steps",
            save_steps=1600,
            logging_strategy="steps",
            logging_steps=160,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=8,
            gradient_accumulation_steps=4,
            num_train_epochs=3,
            learning_rate=1e-5,
            warmup_ratio=0.06,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            save_total_limit=2,
            optim="adamw_torch",
            lr_scheduler_type="cosine",
            report_to="none",
            seed=42,
        )
    
    # Enhanced metrics with AUC-ROC
    def compute_metrics(pred):
        logits, labels = pred
        preds = logits.argmax(axis=1)
        probs = torch.nn.functional.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
        accuracy = accuracy_score(labels, preds)
        
        # AUC-ROC for better evaluation of imbalanced data
        try:
            auc_roc = roc_auc_score(labels, probs)
        except:
            auc_roc = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(labels, preds)
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "specificity": specificity,
            "sensitivity": sensitivity,
            "auc_roc": auc_roc,
        }
    
    # Custom trainer with class weights - FIXED VERSION
    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            """
            Fixed version that accepts any keyword arguments
            """
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.get('logits')
            loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss
    
    # Create trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )
    
    # Train
    print("\n" + "=" * 60)
    print("Starting RoBERTa training...")
    print("Expected time: 1-2 hours on Colab GPU")
    print("Using class weights to handle imbalance")
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
    trainer.save_model("./results_roberta/final_model")
    tokenizer.save_pretrained("./results_roberta/final_model")
    
    # Training time
    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)
    
    print("\n" + "=" * 60)
    print(f"✓ RoBERTa training completed in {hours}h {minutes}m {seconds}s")
    print(f"✓ Model saved to: ./results_roberta/final_model")
    print("=" * 60)
    
    # Save summary
    with open("./results_roberta/training_summary.txt", "w") as f:
        f.write(f"RoBERTa Training Summary\n")
        f.write(f"{'='*50}\n")
        f.write(f"Model: roberta-base\n")
        f.write(f"Parameters: ~125M\n")
        f.write(f"Max sequence length: 256 tokens\n")
        f.write(f"Class weights applied: Yes\n")
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
        print("1. Run: !zip -r results_roberta.zip results_roberta")
        print("2. Run: from google.colab import files")
        print("3. Run: files.download('results_roberta.zip')")
        print("=" * 60)

if __name__ == "__main__":
    main()