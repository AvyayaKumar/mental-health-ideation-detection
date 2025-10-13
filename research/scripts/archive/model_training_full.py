import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast
import torch
from torch.utils.data import Dataset
from transformers import DistilBertForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import time

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
    print("TRAINING ON FULL DATASET")
    print("=" * 50)
    
    start_time = time.time()
    
    csv_path = "mapped_dataset.csv"
    
    # Load FULL dataset (no sampling)
    print("Loading full dataset...")
    df = pd.read_csv(csv_path, engine="python", on_bad_lines="skip")
    print(f"Total samples in dataset: {len(df)}")
    
    # Split the data
    print("Splitting data into train/val/test sets...")
    train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['class'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['class'], random_state=42)
    
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # Tokenize the data
    print("Tokenizing data...")
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    
    train_encodings = tokenizer(
        train_df['text'].tolist(),
        padding=True,
        truncation=True,
        max_length=128,  # Set explicit max length for consistency
        return_tensors="pt"
    )
    
    val_encodings = tokenizer(
        val_df['text'].tolist(),
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    
    # Also tokenize test set for later evaluation
    test_encodings = tokenizer(
        test_df['text'].tolist(),
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    
    # Create datasets
    train_labels = train_df['class'].tolist()
    val_labels = val_df['class'].tolist()
    test_labels = test_df['class'].tolist()
    
    train_dataset = TextClassificationDataset(train_encodings, train_labels)
    val_dataset = TextClassificationDataset(val_encodings, val_labels)
    test_dataset = TextClassificationDataset(test_encodings, test_labels)
    
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
    
    # Training arguments for FULL dataset
    training_args = TrainingArguments(
        output_dir="./results_full",  # Different directory for full model
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,  # Log every 100 steps
        per_device_train_batch_size=16 if device.type == "cuda" else 8,  # Adjust batch size based on device
        per_device_eval_batch_size=32 if device.type == "cuda" else 16,
        num_train_epochs=3,  # You might want to increase this for full dataset
        learning_rate=2e-5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        dataloader_num_workers=4 if device.type == "cuda" else 0,  # Adjust workers based on device
        # max_steps removed - will train on full dataset
        no_cuda=False if device.type == "cuda" else True,
        warmup_steps=500,  # Add warmup for better convergence
        save_total_limit=2,  # Only keep 2 best checkpoints to save space
        report_to="tensorboard",  # Enable tensorboard logging
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
    print("Starting training on FULL dataset...")
    print("This will take significantly longer than the 10% version!")
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
    trainer.save_model("./results_full/final_model")
    tokenizer.save_pretrained("./results_full/final_model")
    
    # Calculate training time
    end_time = time.time()
    training_time = end_time - start_time
    hours = int(training_time // 3600)
    minutes = int((training_time % 3600) // 60)
    seconds = int(training_time % 60)
    
    print("\n" + "=" * 50)
    print(f"Training completed in {hours}h {minutes}m {seconds}s")
    print(f"Model saved to: ./results_full/final_model")
    print("=" * 50)
    
    # Save results to file
    with open("./results_full/training_summary.txt", "w") as f:
        f.write(f"Training on FULL Dataset Summary\n")
        f.write(f"=" * 50 + "\n")
        f.write(f"Total samples: {len(df)}\n")
        f.write(f"Training samples: {len(train_df)}\n")
        f.write(f"Validation samples: {len(val_df)}\n")
        f.write(f"Test samples: {len(test_df)}\n")
        f.write(f"Device used: {device}\n")
        f.write(f"Training time: {hours}h {minutes}m {seconds}s\n")
        f.write(f"\nValidation Results:\n{eval_results}\n")
        f.write(f"\nTest Results:\n{test_results}\n")

if __name__ == "__main__":
    main()