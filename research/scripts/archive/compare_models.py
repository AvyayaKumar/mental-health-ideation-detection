import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def evaluate_model(model_path, tokenizer, test_texts, test_labels, model_name):
    """Evaluate a single model and return metrics"""
    print(f"\nEvaluating {model_name}...")
    
    # Load model
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    model.to(device).eval()
    
    predictions = []
    probabilities = []
    
    # Process in batches for efficiency
    batch_size = 32
    for i in range(0, len(test_texts), batch_size):
        batch_texts = test_texts[i:i+batch_size]
        
        # Tokenize
        enc = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        
        # Predict
        with torch.no_grad():
            outputs = model(**enc)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(test_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(test_labels, predictions, average='binary')
    cm = confusion_matrix(test_labels, predictions)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'predictions': predictions,
        'probabilities': probabilities
    }

def compare_on_examples(model_10_path, model_full_path, tokenizer):
    """Compare models on specific example sentences"""
    examples = [
        "I feel like hurting myself today.",
        "I'm so happy I got a promotion!",
        "Life has no meaning anymore.",
        "I love spending time with my family.",
        "I don't want to be here anymore.",
        "The weather is beautiful today!",
        "I can't take this pain anymore.",
        "Looking forward to the weekend!",
        "Nobody would miss me if I was gone.",
        "Just finished a great workout!"
    ]
    
    # Load both models
    model_10 = DistilBertForSequenceClassification.from_pretrained(model_10_path)
    model_10.to(device).eval()
    
    try:
        model_full = DistilBertForSequenceClassification.from_pretrained(model_full_path)
        model_full.to(device).eval()
        has_full_model = True
    except:
        print("Full model not found. Train it first using model_training_full.py")
        has_full_model = False
        model_full = None
    
    print("\n" + "=" * 80)
    print("EXAMPLE PREDICTIONS COMPARISON")
    print("=" * 80)
    
    for text in examples:
        enc = tokenizer([text], padding=True, truncation=True, max_length=128, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        
        with torch.no_grad():
            # 10% model predictions
            logits_10 = model_10(**enc).logits
            probs_10 = torch.softmax(logits_10, dim=-1)
            pred_10 = torch.argmax(probs_10, dim=1)
            conf_10 = float(probs_10[0][pred_10])
            
            # Full model predictions (if available)
            if has_full_model:
                logits_full = model_full(**enc).logits
                probs_full = torch.softmax(logits_full, dim=-1)
                pred_full = torch.argmax(probs_full, dim=1)
                conf_full = float(probs_full[0][pred_full])
        
        print(f"\nText: \"{text}\"")
        label_10 = "Suicidal" if pred_10 == 1 else "Not Suicidal"
        print(f"  10% Model: {label_10} (confidence: {conf_10:.2%})")
        
        if has_full_model:
            label_full = "Suicidal" if pred_full == 1 else "Not Suicidal"
            print(f"  Full Model: {label_full} (confidence: {conf_full:.2%})")
            
            if pred_10 != pred_full:
                print(f"  ⚠️  MODELS DISAGREE!")

def main():
    # Load tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    
    # Load test data
    print("Loading test data...")
    df = pd.read_csv("mapped_dataset.csv", engine="python", on_bad_lines="skip")
    
    # Use the same split as in training
    from sklearn.model_selection import train_test_split
    _, temp_df = train_test_split(df, test_size=0.3, stratify=df['class'], random_state=42)
    _, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['class'], random_state=42)
    
    test_texts = test_df['text'].tolist()
    test_labels = test_df['class'].tolist()
    
    print(f"Test set size: {len(test_texts)} samples")
    
    # Evaluate 10% model
    results_10 = evaluate_model(
        "results/checkpoint-500",
        tokenizer,
        test_texts,
        test_labels,
        "10% Model"
    )
    
    print("\n" + "=" * 50)
    print("10% MODEL RESULTS")
    print("=" * 50)
    print(f"Accuracy:  {results_10['accuracy']:.4f}")
    print(f"Precision: {results_10['precision']:.4f}")
    print(f"Recall:    {results_10['recall']:.4f}")
    print(f"F1 Score:  {results_10['f1']:.4f}")
    print("\nConfusion Matrix:")
    print(results_10['confusion_matrix'])
    
    # Try to evaluate full model if it exists
    try:
        results_full = evaluate_model(
            "results_full/final_model",
            tokenizer,
            test_texts,
            test_labels,
            "Full Model"
        )
        
        print("\n" + "=" * 50)
        print("FULL MODEL RESULTS")
        print("=" * 50)
        print(f"Accuracy:  {results_full['accuracy']:.4f}")
        print(f"Precision: {results_full['precision']:.4f}")
        print(f"Recall:    {results_full['recall']:.4f}")
        print(f"F1 Score:  {results_full['f1']:.4f}")
        print("\nConfusion Matrix:")
        print(results_full['confusion_matrix'])
        
        # Compare improvements
        print("\n" + "=" * 50)
        print("IMPROVEMENT COMPARISON")
        print("=" * 50)
        print(f"Accuracy:  {results_10['accuracy']:.4f} -> {results_full['accuracy']:.4f} ({(results_full['accuracy'] - results_10['accuracy'])*100:+.2f}%)")
        print(f"Precision: {results_10['precision']:.4f} -> {results_full['precision']:.4f} ({(results_full['precision'] - results_10['precision'])*100:+.2f}%)")
        print(f"Recall:    {results_10['recall']:.4f} -> {results_full['recall']:.4f} ({(results_full['recall'] - results_10['recall'])*100:+.2f}%)")
        print(f"F1 Score:  {results_10['f1']:.4f} -> {results_full['f1']:.4f} ({(results_full['f1'] - results_10['f1'])*100:+.2f}%)")
        
    except Exception as e:
        print("\n" + "=" * 50)
        print("FULL MODEL NOT FOUND")
        print("=" * 50)
        print("To train the full model, run:")
        print("  python model_training_full.py")
    
    # Compare on examples
    compare_on_examples("results/checkpoint-500", "results_full/final_model", tokenizer)

if __name__ == "__main__":
    main()