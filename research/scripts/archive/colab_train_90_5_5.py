"""
Google Colab Training Script - 90/5/5 Split
============================================
This script is optimized for training on Google Colab with the new 90/5/5 split
Copy and run each cell in Google Colab
"""

# ===== CELL 1: Check GPU and Install Dependencies =====
"""
!nvidia-smi  # Check GPU availability
!pip install transformers accelerate torch safetensors scikit-learn pandas tqdm
"""

# ===== CELL 2: Mount Google Drive (Optional - for saving results) =====
"""
from google.colab import drive
drive.mount('/content/drive')
"""

# ===== CELL 3: Upload Dataset =====
"""
from google.colab import files
import os

print("Please upload your 'Suicide_Detection 2.csv' or 'Suicide_Detection_Full.csv' file")
uploaded = files.upload()

# List uploaded files
print("\\nUploaded files:")
for filename in uploaded.keys():
    print(f"  - {filename}")
"""

# ===== CELL 4: Run the Training Script =====
"""
# First, upload the model_training_232k_colab.py file
print("Please upload the 'model_training_232k_colab.py' file")
uploaded = files.upload()

# Run the training
!python model_training_232k_colab.py
"""

# ===== CELL 5: Alternative - Direct Training Code =====
"""
# If you prefer to run the code directly in the notebook:

import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, TrainingArguments, Trainer
import torch
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import time
import gc
import os

# Check GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Dataset class
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

print("=" * 50)
print("TRAINING WITH 90/5/5 SPLIT")
print("=" * 50)

# Load dataset
csv_files = ['Suicide_Detection 2.csv', 'Suicide_Detection_Full.csv', 'Suicide_Detection_2.csv']
csv_path = None
for file in csv_files:
    if os.path.exists(file):
        csv_path = file
        break

if csv_path is None:
    raise FileNotFoundError("Dataset not found! Please upload the CSV file")

print(f"Loading dataset from {csv_path}...")
df = pd.read_csv(csv_path, dtype={'text': str, 'class': str})
df = df.dropna()

# Map labels
class_mapping = {'non-suicide': 0, 'suicide': 1}
df['class'] = df['class'].map(class_mapping)
df = df.dropna()
df['class'] = df['class'].astype(int)

print(f"Total samples: {len(df)}")
print(f"Class distribution:")
print(df['class'].value_counts())

# 90/5/5 split
print("\\nCreating 90/5/5 train/val/test split...")
train_df, temp_df = train_test_split(df, test_size=0.1, stratify=df['class'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['class'], random_state=42)

print(f"Train: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
print(f"Val: {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
print(f"Test: {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")

# Clear memory
del df
gc.collect()
torch.cuda.empty_cache()

# Tokenization
print("\\nTokenizing data...")
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def tokenize_batch(texts, batch_size=5000):
    all_encodings = None
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        if i % 10000 == 0:
            print(f"  Processing {i}/{len(texts)}...")
        
        batch_enc = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
        
        if all_encodings is None:
            all_encodings = batch_enc
        else:
            for key in all_encodings:
                all_encodings[key] = torch.cat([all_encodings[key], batch_enc[key]])
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return all_encodings

train_encodings = tokenize_batch(train_df['text'].tolist())
val_encodings = tokenize_batch(val_df['text'].tolist())
test_encodings = tokenize_batch(test_df['text'].tolist())

# Create datasets
train_dataset = TextClassificationDataset(train_encodings, train_df['class'].tolist())
val_dataset = TextClassificationDataset(val_encodings, val_df['class'].tolist())
test_dataset = TextClassificationDataset(test_encodings, test_df['class'].tolist())

# Clear memory
del train_df, val_df, test_df
gc.collect()
torch.cuda.empty_cache()

# Load model
print("\\nLoading model...")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2).to(device)

# Training arguments optimized for 90/5/5 split
training_args = TrainingArguments(
    output_dir="./results_90_5_5",
    eval_strategy="steps",
    eval_steps=1000,  # More frequent evaluation due to more training data
    save_strategy="steps",
    save_steps=2000,
    logging_steps=100,
    per_device_train_batch_size=16 if torch.cuda.is_available() else 8,
    per_device_eval_batch_size=32 if torch.cuda.is_available() else 16,
    num_train_epochs=2,  # May need fewer epochs due to more training data
    learning_rate=2e-5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    warmup_steps=1000,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),  # Mixed precision if GPU available
    dataloader_pin_memory=True,
    report_to="none",
)

# Metrics
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

# Train
print("\\n" + "=" * 50)
print("Starting training with 90/5/5 split...")
print("Expected time: 45-90 minutes on Colab GPU")
print("=" * 50)

start_time = time.time()
trainer.train()

# Evaluate
print("\\nEvaluating on validation set...")
val_results = trainer.evaluate()
print(f"Validation Results: {val_results}")

print("\\nEvaluating on test set...")
test_results = trainer.evaluate(test_dataset)
print(f"Test Results: {test_results}")

# Save model
print("\\nSaving model...")
trainer.save_model("./results_90_5_5/final_model")
tokenizer.save_pretrained("./results_90_5_5/final_model")

# Training time
elapsed = time.time() - start_time
print(f"\\nTraining completed in {elapsed/3600:.2f} hours")

# Save summary
with open("./results_90_5_5/training_summary.txt", "w") as f:
    f.write(f"Training with 90/5/5 Split\\n")
    f.write(f"{'='*50}\\n")
    f.write(f"Train samples: {len(train_dataset)}\\n")
    f.write(f"Val samples: {len(val_dataset)}\\n")
    f.write(f"Test samples: {len(test_dataset)}\\n")
    f.write(f"Training time: {elapsed/3600:.2f} hours\\n")
    f.write(f"\\nValidation Results:\\n{val_results}\\n")
    f.write(f"\\nTest Results:\\n{test_results}\\n")

print("Training complete! Model saved to ./results_90_5_5/final_model")
"""

# ===== CELL 6: Download Results =====
"""
# Zip the results
!zip -r results_90_5_5.zip results_90_5_5/

# Download
from google.colab import files
files.download('results_90_5_5.zip')

print("Results downloaded! You can now use the trained model locally.")
"""

# ===== CELL 7: Quick Test of Trained Model =====
"""
# Test the trained model with sample text
from transformers import pipeline

# Load the trained model
model_path = "./results_90_5_5/final_model"
classifier = pipeline("text-classification", model=model_path, device=0 if torch.cuda.is_available() else -1)

# Test samples
test_texts = [
    "I'm feeling great today!",
    "Life is wonderful and full of opportunities",
    "I'm struggling but trying to stay positive",
]

print("Testing trained model:\\n")
for text in test_texts:
    result = classifier(text)[0]
    label = "suicide" if result['label'] == 'LABEL_1' else "non-suicide"
    confidence = result['score']
    print(f"Text: '{text[:50]}...'")
    print(f"  Prediction: {label} (confidence: {confidence:.2%})\\n")
"""

print("Colab training script created successfully!")
print("\nInstructions:")
print("1. Open Google Colab (colab.research.google.com)")
print("2. Create a new notebook")
print("3. Copy and run each cell from this script")
print("4. Upload your dataset when prompted")
print("5. The model will train with 90/5/5 splits")
print("\nKey changes from 60/20/20 to 90/5/5:")
print("- More training data (90% vs 60%) for better learning")
print("- Smaller validation/test sets (5% each vs 20% each)")
print("- This should improve model accuracy with more training examples")