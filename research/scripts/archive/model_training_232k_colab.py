"""
Google Colab optimized training script for 232K dataset
Run this in Colab after uploading your 'Suicide_Detection 2.csv' file
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast
import torch
from torch.utils.data import Dataset
from transformers import DistilBertForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import time
import gc
import os
import warnings
warnings.filterwarnings('ignore')

# Check if we're in Colab and if GPU is available
try:
    import google.colab
    IN_COLAB = True
    print("Running in Google Colab")
    
    # Enable GPU if available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device("cpu")
        print("⚠️ No GPU detected, using CPU (will be slower)")
except:
    IN_COLAB = False
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Running locally on {device}")

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
    print("=" * 50)
    print("TRAINING ON 232K DATASET (Google Colab)")
    print("=" * 50)
    
    start_time = time.time()
    
    # Check for the CSV file - try multiple possible names
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
        print("You can drag and drop it into the file browser on the left")
        return
    
    print(f"Loading 232K dataset from {csv_path}...")
    
    # Read CSV
    df = pd.read_csv(csv_path, dtype={'text': str, 'class': str})
    
    # Clean the data
    print("Cleaning data...")
    df = df.dropna()
    
    # Map class labels to integers
    class_mapping = {'non-suicide': 0, 'suicide': 1}
    df['class'] = df['class'].map(class_mapping)
    df = df.dropna()
    df['class'] = df['class'].astype(int)
    
    print(f"Dataset loaded: {len(df)} samples")
    
    # Check class distribution
    class_dist = df['class'].value_counts()
    print(f"\nClass distribution:")
    print(f"  Non-suicide (0): {class_dist.get(0, 0)} samples")
    print(f"  Suicide (1): {class_dist.get(1, 0)} samples")
    
    # Split the data (90/5/5 split)
    print("\nSplitting data into train/val/test sets...")
    train_df, temp_df = train_test_split(df, test_size=0.1, stratify=df['class'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['class'], random_state=42)
    
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # Clear memory
    del df
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Tokenize the data
    print("\nTokenizing data (this will take a few minutes)...")
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    
    # Process in smaller chunks to avoid memory issues
    def tokenize_in_batches(texts, batch_size=5000):
        all_encodings = None
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            if i % (batch_size * 5) == 0:  # Print every 5 batches
                print(f"  Tokenizing batch {i//batch_size + 1}/{total_batches}...")
            
            batch_encodings = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )
            
            if all_encodings is None:
                all_encodings = batch_encodings
            else:
                for key in all_encodings:
                    all_encodings[key] = torch.cat([all_encodings[key], batch_encodings[key]])
            
            # Clear GPU cache periodically
            if torch.cuda.is_available() and i % (batch_size * 10) == 0:
                torch.cuda.empty_cache()
        
        return all_encodings
    
    print("Tokenizing training data...")
    train_encodings = tokenize_in_batches(train_df['text'].tolist())
    
    print("Tokenizing validation data...")
    val_encodings = tokenize_in_batches(val_df['text'].tolist())
    
    print("Tokenizing test data...")
    test_encodings = tokenize_in_batches(test_df['text'].tolist())
    
    # Create datasets
    train_labels = train_df['class'].tolist()
    val_labels = val_df['class'].tolist()
    test_labels = test_df['class'].tolist()
    
    train_dataset = TextClassificationDataset(train_encodings, train_labels)
    val_dataset = TextClassificationDataset(val_encodings, val_labels)
    test_dataset = TextClassificationDataset(test_encodings, test_labels)
    
    # Clear memory
    del train_df, val_df, test_df
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Load the model
    print(f"\nLoading model and moving to {device}...")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2
    ).to(device)
    
    # Training arguments optimized for Colab
    if IN_COLAB and torch.cuda.is_available():
        # Optimal settings for Colab GPU
        training_args = TrainingArguments(
            output_dir="./results_230k",
            eval_strategy="steps",
            eval_steps=2000,
            save_strategy="steps",
            save_steps=2000,
            logging_strategy="steps",
            logging_steps=200,
            per_device_train_batch_size=16,  # Colab GPU can handle larger batches
            per_device_eval_batch_size=32,
            gradient_accumulation_steps=1,
            num_train_epochs=2,
            learning_rate=2e-5,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            dataloader_num_workers=2,
            warmup_steps=1000,
            save_total_limit=2,
            fp16=True,  # Enable mixed precision on GPU
            dataloader_pin_memory=True,
            report_to="none",  # Disable wandb/tensorboard in Colab
        )
    else:
        # Settings for CPU or non-Colab
        training_args = TrainingArguments(
            output_dir="./results_230k",
            eval_strategy="steps",
            eval_steps=5000,
            save_strategy="steps",
            save_steps=5000,
            logging_strategy="steps",
            logging_steps=500,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=8,
            gradient_accumulation_steps=2,
            num_train_epochs=2,
            learning_rate=2e-5,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            dataloader_num_workers=0,
            warmup_steps=1000,
            save_total_limit=2,
            report_to="none",
        )
    
    # Metrics computation function
    def compute_metrics(pred):
        logits, labels = pred
        preds = logits.argmax(axis=1)
        p, r, f, _ = precision_recall_fscore_support(labels, preds, average="binary")
        a = accuracy_score(labels, preds)
        return {"accuracy": a, "precision": p, "recall": r, "f1": f}
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )
    
    # Train the model
    print("\n" + "=" * 50)
    print("Starting training on 232K DATASET...")
    if IN_COLAB and torch.cuda.is_available():
        print("Using GPU - this should take 30-60 minutes")
    else:
        print("Using CPU - this will take several hours")
    print("=" * 50 + "\n")
    
    trainer.train()
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    eval_results = trainer.evaluate()
    print(f"Validation Results: {eval_results}")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = trainer.evaluate(test_dataset)
    print(f"Test Results: {test_results}")
    
    # Save the final model
    print("\nSaving final model...")
    trainer.save_model("./results_230k/final_model")
    tokenizer.save_pretrained("./results_230k/final_model")
    
    # Calculate training time
    end_time = time.time()
    training_time = end_time - start_time
    hours = int(training_time // 3600)
    minutes = int((training_time % 3600) // 60)
    seconds = int(training_time % 60)
    
    print("\n" + "=" * 50)
    print(f"✓ Training completed in {hours}h {minutes}m {seconds}s")
    print(f"✓ Model saved to: ./results_230k/final_model")
    print("=" * 50)
    
    # Save results to file
    with open("./results_230k/training_summary.txt", "w") as f:
        f.write(f"Training on 232K Dataset Summary\n")
        f.write(f"=" * 50 + "\n")
        f.write(f"Total samples: {len(train_labels) + len(val_labels) + len(test_labels)}\n")
        f.write(f"Training samples: {len(train_labels)}\n")
        f.write(f"Validation samples: {len(val_labels)}\n")
        f.write(f"Test samples: {len(test_labels)}\n")
        f.write(f"Device used: {device}\n")
        f.write(f"Training time: {hours}h {minutes}m {seconds}s\n")
        f.write(f"\nValidation Results:\n{eval_results}\n")
        f.write(f"\nTest Results:\n{test_results}\n")
    
    print("\nTraining summary saved to ./results_230k/training_summary.txt")
    
    if IN_COLAB:
        print("\n" + "=" * 50)
        print("NEXT STEPS:")
        print("1. Run: !zip -r results_232k.zip results_230k")
        print("2. Run: from google.colab import files")
        print("3. Run: files.download('results_232k.zip')")
        print("=" * 50)

if __name__ == "__main__":
    main()