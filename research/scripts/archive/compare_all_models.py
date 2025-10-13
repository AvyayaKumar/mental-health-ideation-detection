import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
import os

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def evaluate_model(model_path, tokenizer, test_texts, test_labels, model_name):
    """Evaluate a single model and return metrics"""
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return None
    
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

def compare_on_examples(models_dict, tokenizer):
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
        "Just finished a great workout!",
        "Everything feels hopeless.",
        "Excited about the new project!",
    ]
    
    # Load all available models
    loaded_models = {}
    for name, path in models_dict.items():
        if os.path.exists(path):
            model = DistilBertForSequenceClassification.from_pretrained(path)
            model.to(device).eval()
            loaded_models[name] = model
            print(f"✓ Loaded {name}")
        else:
            print(f"✗ {name} not found at {path}")
    
    if not loaded_models:
        print("No models found to compare!")
        return
    
    print("\n" + "=" * 80)
    print("EXAMPLE PREDICTIONS COMPARISON")
    print("=" * 80)
    
    for text in examples:
        enc = tokenizer([text], padding=True, truncation=True, max_length=128, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        
        print(f"\nText: \"{text}\"")
        
        predictions = {}
        for name, model in loaded_models.items():
            with torch.no_grad():
                logits = model(**enc).logits
                probs = torch.softmax(logits, dim=-1)
                pred = torch.argmax(probs, dim=1)
                conf = float(probs[0][pred])
                
                label = "Suicidal" if pred == 1 else "Not Suicidal"
                predictions[name] = (label, conf)
                print(f"  {name:15s}: {label:15s} (confidence: {conf:.2%})")
        
        # Check if models disagree
        unique_predictions = set([p[0] for p in predictions.values()])
        if len(unique_predictions) > 1:
            print(f"  ⚠️  MODELS DISAGREE!")

def main():
    # Model configurations
    models = {
        "10% Model": "results/checkpoint-500",
        "Full Model": "results_full/final_model",
        "230K Model": "results_230k/final_model"
    }
    
    # Load tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    
    # Load test data from the original dataset for fair comparison
    print("Loading test data...")
    df = pd.read_csv("mapped_dataset.csv", engine="python", on_bad_lines="skip")
    
    # Use the same split as in training
    from sklearn.model_selection import train_test_split
    _, temp_df = train_test_split(df, test_size=0.3, stratify=df['class'], random_state=42)
    _, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['class'], random_state=42)
    
    test_texts = test_df['text'].tolist()
    test_labels = test_df['class'].tolist()
    
    print(f"Test set size: {len(test_texts)} samples")
    
    # Store results
    all_results = {}
    
    # Evaluate each model
    for model_name, model_path in models.items():
        results = evaluate_model(
            model_path,
            tokenizer,
            test_texts,
            test_labels,
            model_name
        )
        if results:
            all_results[model_name] = results
    
    # Display results
    print("\n" + "=" * 70)
    print("MODEL PERFORMANCE COMPARISON")
    print("=" * 70)
    print(f"{'Model':<15} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1 Score':<10}")
    print("-" * 70)
    
    for model_name, results in all_results.items():
        print(f"{model_name:<15} {results['accuracy']:.4f}     {results['precision']:.4f}      {results['recall']:.4f}     {results['f1']:.4f}")
    
    # Show improvements if we have multiple models
    if len(all_results) > 1:
        print("\n" + "=" * 70)
        print("IMPROVEMENTS FROM BASE MODEL (10%)")
        print("=" * 70)
        
        if "10% Model" in all_results:
            base_results = all_results["10% Model"]
            
            for model_name, results in all_results.items():
                if model_name != "10% Model":
                    print(f"\n{model_name}:")
                    acc_diff = (results['accuracy'] - base_results['accuracy']) * 100
                    prec_diff = (results['precision'] - base_results['precision']) * 100
                    rec_diff = (results['recall'] - base_results['recall']) * 100
                    f1_diff = (results['f1'] - base_results['f1']) * 100
                    
                    print(f"  Accuracy:  {acc_diff:+.2f}%")
                    print(f"  Precision: {prec_diff:+.2f}%")
                    print(f"  Recall:    {rec_diff:+.2f}%")
                    print(f"  F1 Score:  {f1_diff:+.2f}%")
    
    # Compare on examples
    compare_on_examples(models, tokenizer)
    
    # Save comparison results
    with open("model_comparison_results.txt", "w") as f:
        f.write("MODEL PERFORMANCE COMPARISON\n")
        f.write("=" * 70 + "\n")
        f.write(f"Test set size: {len(test_texts)} samples\n\n")
        
        for model_name, results in all_results.items():
            f.write(f"{model_name}:\n")
            f.write(f"  Accuracy:  {results['accuracy']:.4f}\n")
            f.write(f"  Precision: {results['precision']:.4f}\n")
            f.write(f"  Recall:    {results['recall']:.4f}\n")
            f.write(f"  F1 Score:  {results['f1']:.4f}\n")
            f.write(f"  Confusion Matrix:\n")
            f.write(f"    {results['confusion_matrix'][0]}\n")
            f.write(f"    {results['confusion_matrix'][1]}\n\n")
    
    print("\n\nResults saved to model_comparison_results.txt")

if __name__ == "__main__":
    main()