"""
Google Colab Script - Train & Compare All Models
================================================
This script trains and compares 4 different models for suicide ideation detection:
1. DistilBERT (baseline - fastest)
2. BERT-base (more accurate than DistilBERT)
3. RoBERTa (optimized for classification)
4. XLNet (best for nuanced understanding)

All models use 90/5/5 train/val/test split
"""

# ========== CELL 1: Setup & Install ==========
"""
!nvidia-smi
!pip install transformers accelerate torch safetensors scikit-learn pandas tqdm matplotlib seaborn
"""

# ========== CELL 2: Upload Dataset ==========
"""
from google.colab import files
import os

print("Please upload your 'Suicide_Detection 2.csv' file")
uploaded = files.upload()

for filename in uploaded.keys():
    print(f"Uploaded: {filename}")
"""

# ========== CELL 3: Upload Training Scripts ==========
"""
print("Please upload the following training scripts:")
print("1. model_training_232k_colab.py (DistilBERT)")
print("2. train_bert_base.py")
print("3. train_roberta.py")
print("4. train_xlnet.py")

uploaded = files.upload()
for filename in uploaded.keys():
    print(f"Uploaded: {filename}")
"""

# ========== CELL 4: Train All Models ==========
"""
import time
import os

models_to_train = [
    ("DistilBERT", "model_training_232k_colab.py"),
    ("BERT-base", "train_bert_base.py"),
    ("RoBERTa", "train_roberta.py"),
    ("XLNet", "train_xlnet.py")
]

training_times = {}

print("=" * 60)
print("TRAINING ALL MODELS")
print("Total expected time: 4-6 hours on GPU")
print("=" * 60)

for model_name, script_name in models_to_train:
    if os.path.exists(script_name):
        print(f"\\n{'='*60}")
        print(f"Training {model_name}...")
        print(f"{'='*60}")
        
        start = time.time()
        !python {script_name}
        elapsed = time.time() - start
        
        training_times[model_name] = elapsed
        print(f"\\n✓ {model_name} completed in {elapsed/3600:.2f} hours")
    else:
        print(f"⚠️ Skipping {model_name} - {script_name} not found")

print("\\n" + "=" * 60)
print("ALL MODELS TRAINED")
print("Training times:")
for model, time_taken in training_times.items():
    print(f"  {model}: {time_taken/3600:.2f} hours")
print("=" * 60)
"""

# ========== CELL 5: Compare All Models ==========
"""
import pandas as pd
import torch
from transformers import (
    DistilBertTokenizerFast, DistilBertForSequenceClassification,
    BertTokenizerFast, BertForSequenceClassification,
    RobertaTokenizerFast, RobertaForSequenceClassification,
    XLNetTokenizerFast, XLNetForSequenceClassification,
    pipeline
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model configurations
models_config = [
    {
        "name": "DistilBERT",
        "path": "./results_230k/final_model",
        "tokenizer": DistilBertTokenizerFast,
        "model_class": DistilBertForSequenceClassification
    },
    {
        "name": "BERT-base",
        "path": "./results_bert_base/final_model",
        "tokenizer": BertTokenizerFast,
        "model_class": BertForSequenceClassification
    },
    {
        "name": "RoBERTa",
        "path": "./results_roberta/final_model",
        "tokenizer": RobertaTokenizerFast,
        "model_class": RobertaForSequenceClassification
    },
    {
        "name": "XLNet",
        "path": "./results_xlnet/final_model",
        "tokenizer": XLNetTokenizerFast,
        "model_class": XLNetForSequenceClassification
    }
]

# Load test data
print("Loading test data...")
csv_files = ['Suicide_Detection 2.csv', 'Suicide_Detection_Full.csv']
csv_path = None
for file in csv_files:
    if os.path.exists(file):
        csv_path = file
        break

df = pd.read_csv(csv_path, dtype={'text': str, 'class': str})
df = df.dropna()

# Map labels
class_mapping = {'non-suicide': 0, 'suicide': 1}
df['class'] = df['class'].map(class_mapping)
df = df.dropna()
df['class'] = df['class'].astype(int)

# Get test split (matching the 90/5/5 split)
from sklearn.model_selection import train_test_split
_, temp_df = train_test_split(df, test_size=0.1, stratify=df['class'], random_state=42)
_, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['class'], random_state=42)

# Sample for faster evaluation
test_sample = test_df.sample(n=min(2000, len(test_df)), random_state=42)
print(f"Evaluating on {len(test_sample)} test samples")

# Evaluate each model
results = []

for config in models_config:
    if os.path.exists(config["path"]):
        print(f"\\nEvaluating {config['name']}...")
        
        # Load model and tokenizer
        tokenizer = config["tokenizer"].from_pretrained(config["path"])
        model = config["model_class"].from_pretrained(config["path"]).to(device)
        model.eval()
        
        # Create pipeline
        classifier = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Predict
        predictions = []
        true_labels = test_sample['class'].tolist()
        
        for text in test_sample['text'].tolist():
            try:
                result = classifier(text, truncation=True, max_length=256)[0]
                pred_label = 1 if result['label'] == 'LABEL_1' else 0
                predictions.append(pred_label)
            except:
                predictions.append(0)  # Default to non-suicide if error
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='binary'
        )
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        tn, fp, fn, tp = cm.ravel()
        
        # Store results
        results.append({
            "Model": config["name"],
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "True Positives": tp,
            "False Positives": fp,
            "True Negatives": tn,
            "False Negatives": fn,
            "FNR": fn / (fn + tp) if (fn + tp) > 0 else 0
        })
        
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  False Negative Rate: {fn / (fn + tp) if (fn + tp) > 0 else 0:.4f}")
        
        # Clear memory
        del model, tokenizer, classifier
        torch.cuda.empty_cache()
    else:
        print(f"⚠️ {config['name']} model not found at {config['path']}")

# Create comparison dataframe
results_df = pd.DataFrame(results)
print("\\n" + "=" * 60)
print("MODEL COMPARISON RESULTS")
print("=" * 60)
print(results_df.to_string(index=False))

# Save results
results_df.to_csv("model_comparison_results.csv", index=False)
print("\\nResults saved to model_comparison_results.csv")
"""

# ========== CELL 6: Visualize Results ==========
"""
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Accuracy Comparison
ax1 = axes[0, 0]
models = results_df['Model'].tolist()
accuracies = results_df['Accuracy'].tolist()
bars1 = ax1.bar(models, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.set_ylim([min(accuracies) * 0.95, 1.0])
for bar, acc in zip(bars1, accuracies):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
             f'{acc:.3f}', ha='center', va='bottom', fontsize=10)

# 2. F1 Score Comparison
ax2 = axes[0, 1]
f1_scores = results_df['F1 Score'].tolist()
bars2 = ax2.bar(models, f1_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
ax2.set_title('F1 Score Comparison', fontsize=14, fontweight='bold')
ax2.set_ylabel('F1 Score', fontsize=12)
ax2.set_ylim([min(f1_scores) * 0.95, 1.0])
for bar, f1 in zip(bars2, f1_scores):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
             f'{f1:.3f}', ha='center', va='bottom', fontsize=10)

# 3. Precision vs Recall
ax3 = axes[1, 0]
precisions = results_df['Precision'].tolist()
recalls = results_df['Recall'].tolist()
x = np.arange(len(models))
width = 0.35
bars3_1 = ax3.bar(x - width/2, precisions, width, label='Precision', color='#1f77b4')
bars3_2 = ax3.bar(x + width/2, recalls, width, label='Recall', color='#ff7f0e')
ax3.set_title('Precision vs Recall', fontsize=14, fontweight='bold')
ax3.set_ylabel('Score', fontsize=12)
ax3.set_xticks(x)
ax3.set_xticklabels(models)
ax3.legend()
ax3.set_ylim([0, 1.05])

# 4. False Negative Rate (Critical for Suicide Detection)
ax4 = axes[1, 1]
fnr_values = results_df['FNR'].tolist()
bars4 = ax4.bar(models, fnr_values, color=['#d62728', '#d62728', '#d62728', '#d62728'])
ax4.set_title('False Negative Rate (Lower is Better)', fontsize=14, fontweight='bold')
ax4.set_ylabel('False Negative Rate', fontsize=12)
ax4.set_ylim([0, max(fnr_values) * 1.2])
for bar, fnr in zip(bars4, fnr_values):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
             f'{fnr:.3f}', ha='center', va='bottom', fontsize=10)
ax4.axhline(y=0.05, color='green', linestyle='--', alpha=0.5, label='Target (<5%)')
ax4.legend()

plt.suptitle('Suicide Ideation Detection Model Comparison', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('model_comparison_plots.png', dpi=300, bbox_inches='tight')
plt.show()

print("Visualization saved as model_comparison_plots.png")
"""

# ========== CELL 7: Recommendation ==========
"""
print("\\n" + "=" * 60)
print("MODEL RECOMMENDATIONS")
print("=" * 60)

# Find best model for different criteria
best_accuracy = results_df.loc[results_df['Accuracy'].idxmax()]
best_f1 = results_df.loc[results_df['F1 Score'].idxmax()]
best_fnr = results_df.loc[results_df['FNR'].idxmin()]

print(f"\\n🏆 BEST OVERALL (F1 Score): {best_f1['Model']}")
print(f"   F1 Score: {best_f1['F1 Score']:.4f}")
print(f"   Accuracy: {best_f1['Accuracy']:.4f}")

print(f"\\n🎯 HIGHEST ACCURACY: {best_accuracy['Model']}")
print(f"   Accuracy: {best_accuracy['Accuracy']:.4f}")

print(f"\\n⚠️ LOWEST FALSE NEGATIVE RATE: {best_fnr['Model']}")
print(f"   FNR: {best_fnr['FNR']:.4f}")
print(f"   (Critical for not missing actual suicide ideation cases)")

print("\\n📊 Summary:")
print("• For production use: Consider the model with lowest FNR")
print("• For research: F1 score provides best balance")
print("• DistilBERT: Fastest inference, good for real-time")
print("• BERT/RoBERTa/XLNet: Higher accuracy but slower")

# Speed comparison (approximate)
print("\\n⚡ Inference Speed (relative):")
print("  DistilBERT: 1x (fastest)")
print("  BERT-base:  ~0.6x")
print("  RoBERTa:    ~0.5x")
print("  XLNet:      ~0.4x (slowest)")
"""

# ========== CELL 8: Download All Results ==========
"""
!zip -r all_model_results.zip results_230k/ results_bert_base/ results_roberta/ results_xlnet/ *.csv *.png *.txt

from google.colab import files
files.download('all_model_results.zip')

print("\\n✅ All model results downloaded!")
print("\\nContents:")
print("• Trained models for DistilBERT, BERT, RoBERTa, XLNet")
print("• Comparison results CSV")
print("• Visualization plots")
print("• Training summaries")
"""