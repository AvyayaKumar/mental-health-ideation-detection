"""
Evaluation script for the 232K model in Google Colab
Run this after training the 232K model
"""

import pandas as pd
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import os
import numpy as np

# Check device
try:
    import google.colab
    IN_COLAB = True
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Running in Colab on {device}")
except:
    IN_COLAB = False
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Running locally on {device}")

# Dataset class
class TextClassificationDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

def evaluate_model():
    print("=" * 60)
    print("EVALUATING 232K MODEL")
    print("=" * 60)
    
    # Check if model exists
    model_path = "results_230k/final_model"
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        print("Make sure you've trained the model first using model_training_232k_colab.py")
        return
    
    # Load test data - try to find the dataset
    csv_files = ['Suicide_Detection 2.csv', 'Suicide_Detection_Full.csv', 'Suicide_Detection_2.csv']
    csv_path = None
    
    for file in csv_files:
        if os.path.exists(file):
            csv_path = file
            break
    
    if csv_path is None:
        print("ERROR: Dataset not found!")
        print("Please upload the original dataset file")
        return
    
    print(f"Loading test data from {csv_path}...")
    
    # Load the same dataset used for training
    df = pd.read_csv(csv_path, dtype={'text': str, 'class': str})
    df = df.dropna()
    
    # Map class labels
    class_mapping = {'non-suicide': 0, 'suicide': 1}
    df['class'] = df['class'].map(class_mapping)
    df = df.dropna()
    df['class'] = df['class'].astype(int)
    
    print(f"Dataset loaded: {len(df)} samples")
    
    # Use the SAME split as training (important for fair evaluation)
    _, temp_df = train_test_split(df, test_size=0.1, stratify=df['class'], random_state=42)
    _, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['class'], random_state=42)
    
    print(f"Test set size: {len(test_df)} samples")
    
    # For faster evaluation, you can sample a subset
    # Uncomment the next line to evaluate on 5000 samples instead of all ~46k
    # test_df = test_df.sample(n=5000, random_state=42)
    
    print(f"Evaluating on {len(test_df)} samples")
    
    # Tokenize test data
    print("Tokenizing test data...")
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    
    # Process in batches to avoid memory issues
    def tokenize_texts(texts, batch_size=2000):
        all_encodings = None
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            print(f"  Tokenizing batch {i//batch_size + 1}/{total_batches}")
            
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
    
    test_encodings = tokenize_texts(test_df["text"].tolist())
    test_labels = test_df["class"].tolist()
    test_dataset = TextClassificationDataset(test_encodings, test_labels)
    
    # Load the trained model
    print(f"Loading model from {model_path}...")
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    model.to(device).eval()
    
    # Create data loader
    batch_size = 32 if device.type == "cuda" else 16
    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Inference
    print(f"Running evaluation on {device}...")
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            preds = logits.argmax(dim=1)
            
            # Store results
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary"
    )
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Print results
    print("\n" + "=" * 60)
    print("232K MODEL EVALUATION RESULTS")
    print("=" * 60)
    print(f"Test samples evaluated: {len(all_labels)}")
    print(f"Accuracy : {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"              0      1")
    print(f"Actual   0  {cm[0,0]:6d} {cm[0,1]:6d}")
    print(f"         1  {cm[1,0]:6d} {cm[1,1]:6d}")
    
    # Per-class metrics
    print(f"\nDetailed Classification Report:")
    class_names = ['Non-Suicide', 'Suicide']
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Confidence analysis
    probs_array = np.array(all_probs)
    pred_probs = probs_array[np.arange(len(all_preds)), all_preds]
    
    print(f"\nConfidence Analysis:")
    print(f"Average confidence: {np.mean(pred_probs):.4f}")
    print(f"High confidence (>0.9): {np.sum(pred_probs > 0.9) / len(pred_probs) * 100:.1f}%")
    print(f"Low confidence (<0.6): {np.sum(pred_probs < 0.6) / len(pred_probs) * 100:.1f}%")
    
    # Save results
    results_text = f"""
232K Model Evaluation Results
============================
Test samples: {len(all_labels)}
Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)
Precision: {precision:.4f}
Recall: {recall:.4f}
F1 Score: {f1:.4f}

Confusion Matrix:
{cm}

Average Confidence: {np.mean(pred_probs):.4f}
High Confidence (>0.9): {np.sum(pred_probs > 0.9) / len(pred_probs) * 100:.1f}%
Low Confidence (<0.6): {np.sum(pred_probs < 0.6) / len(pred_probs) * 100:.1f}%
"""
    
    with open("evaluation_results_232k.txt", "w") as f:
        f.write(results_text)
    
    print(f"\n✓ Results saved to 'evaluation_results_232k.txt'")
    
    # Test on some examples
    print(f"\n" + "=" * 60)
    print("EXAMPLE PREDICTIONS")
    print("=" * 60)
    
    test_examples = [
        "I feel like hurting myself today.",
        "I'm so happy I got a promotion!",
        "Life has no meaning anymore.",
        "I love spending time with my family.",
        "I don't want to be here anymore.",
        "The weather is beautiful today!"
    ]
    
    for text in test_examples:
        # Tokenize
        enc = tokenizer([text], padding=True, truncation=True, max_length=128, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        
        # Predict
        with torch.no_grad():
            outputs = model(**enc)
            probs = torch.softmax(outputs.logits, dim=-1)
            pred = torch.argmax(probs, dim=1)
            confidence = float(probs[0][pred])
        
        label = "Suicidal" if pred == 1 else "Not Suicidal"
        print(f"'{text}' → {label} (confidence: {confidence:.2%})")

if __name__ == "__main__":
    evaluate_model()