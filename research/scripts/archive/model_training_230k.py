import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast
import torch
from torch.utils.data import Dataset
from transformers import DistilBertForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import time
import gc
import warnings
warnings.filterwarnings('ignore')

device = (
    torch.device("cuda") if torch.cuda.is_available()   # Colab's GPU  
    else torch.device("mps") if torch.backends.mps.is_available()  # Apple Silicon  
    else torch.device("cpu")  
)

print(f"Using device: {device}")

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
    print("TRAINING ON 232K DATASET")
    print("=" * 50)
    
    start_time = time.time()
    
    # Use the full 230k dataset
    csv_path = "Suicide_Detection_Full.csv"
    
    print("Loading 232K dataset...")
    print("This may take a moment...")
    
    # Read CSV with specific dtype to avoid mixed types
    df = pd.read_csv(csv_path, dtype={'text': str, 'class': str})
    
    # Clean the data
    print("Cleaning data...")
    # Remove any rows with NaN values
    df = df.dropna()
    
    # Map class labels to integers
    class_mapping = {'non-suicide': 0, 'suicide': 1}
    df['class'] = df['class'].map(class_mapping)
    
    # Remove any rows that couldn't be mapped
    df = df.dropna()
    df['class'] = df['class'].astype(int)
    
    # Remove duplicates to improve training
    initial_size = len(df)
    df = df.drop_duplicates(subset=['text'])
    final_size = len(df)
    
    print(f"Dataset loaded: {initial_size} rows")
    print(f"After removing duplicates: {final_size} rows")
    print(f"Removed {initial_size - final_size} duplicate entries")
    
    # For very large datasets, you might want to sample if memory is an issue
    # Uncomment the following line to use a subset (e.g., 100k samples)
    # df = df.sample(n=min(100000, len(df)), random_state=42)
    
    print(f"Final dataset size: {len(df)} samples")
    
    # Check class distribution
    class_dist = df['class'].value_counts()
    print(f"\nClass distribution:")
    print(f"  Non-suicide (0): {class_dist.get(0, 0)} samples")
    print(f"  Suicide (1): {class_dist.get(1, 0)} samples")
    
    # Split the data (90/5/5 split for large dataset)
    print("\nSplitting data into train/val/test sets...")
    train_df, temp_df = train_test_split(df, test_size=0.1, stratify=df['class'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['class'], random_state=42)
    
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # Clear memory
    del df
    gc.collect()
    
    # Tokenize the data
    print("\nTokenizing data (this will take several minutes)...")
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    
    # Process in smaller chunks to avoid memory issues
    def tokenize_in_batches(texts, batch_size=5000):
        all_encodings = None
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
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
    
    print(f"\nDataset sizes:")
    print(f"  Training: {len(train_dataset)}")
    print(f"  Validation: {len(val_dataset)}")
    print(f"  Test: {len(test_dataset)}")
    
    # Load the model
    print(f"\nLoading model and moving to {device}...")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2
    ).to(device)
    
    # Training arguments optimized for very large dataset
    training_args = TrainingArguments(
        output_dir="./results_230k",  # Different directory for 230k model
        eval_strategy="steps",  # Evaluate more frequently
        eval_steps=5000,  # Evaluate every 5000 steps
        save_strategy="steps",
        save_steps=5000,  # Save every 5000 steps
        logging_strategy="steps",
        logging_steps=500,  # Log every 500 steps
        per_device_train_batch_size=8 if device.type in ["cuda", "mps"] else 4,  # Smaller batch for memory
        per_device_eval_batch_size=16 if device.type in ["cuda", "mps"] else 8,
        gradient_accumulation_steps=2,  # Accumulate gradients for effective larger batch
        num_train_epochs=2,  # Fewer epochs for large dataset
        learning_rate=2e-5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        dataloader_num_workers=2 if device.type in ["cuda", "mps"] else 0,
        no_cuda=False if device.type == "cuda" else True,
        warmup_steps=1000,  # More warmup for large dataset
        save_total_limit=3,  # Keep only 3 best checkpoints
        report_to="tensorboard",
        fp16=True if device.type == "cuda" else False,  # Mixed precision for faster training on GPU
        dataloader_pin_memory=True if device.type == "cuda" else False,
        gradient_checkpointing=True,  # Save memory at cost of speed
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
    print("Starting training on 230K+ DATASET...")
    print("This will take several hours!")
    print("Monitor progress in tensorboard: tensorboard --logdir=./results_230k")
    print("=" * 50 + "\n")
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current model...")
        trainer.save_model("./results_230k/interrupted_model")
        tokenizer.save_pretrained("./results_230k/interrupted_model")
    
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
    print(f"Training completed in {hours}h {minutes}m {seconds}s")
    print(f"Model saved to: ./results_230k/final_model")
    print("=" * 50)
    
    # Save results to file
    with open("./results_230k/training_summary.txt", "w") as f:
        f.write(f"Training on 230K+ Dataset Summary\n")
        f.write(f"=" * 50 + "\n")
        f.write(f"Total samples processed: {final_size}\n")
        f.write(f"Training samples: {len(train_labels)}\n")
        f.write(f"Validation samples: {len(val_labels)}\n")
        f.write(f"Test samples: {len(test_labels)}\n")
        f.write(f"Device used: {device}\n")
        f.write(f"Training time: {hours}h {minutes}m {seconds}s\n")
        f.write(f"\nValidation Results:\n{eval_results}\n")
        f.write(f"\nTest Results:\n{test_results}\n")
    
    print("\nTraining summary saved to ./results_230k/training_summary.txt")

if __name__ == "__main__":
    main()