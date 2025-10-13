import pandas as pd
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm # Progress bar for loops

# Force CPU only
# May cause issues on Apple Silicon or other machines (but works on Colab)
device = torch.device("cpu")

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

# Load & sample test set
csv_path = "mapped_dataset.csv"
df = (
    pd.read_csv(csv_path, engine="python", on_bad_lines="skip")
      .sample(frac=0.1, random_state=42)
)

# Original 70/15/15 split
# Here we only need the test split
_, temp_df = train_test_split(df, test_size=0.3, stratify=df["class"], random_state=42)
_, full_test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["class"], random_state=42)

# Take just 1000 samples for a quick run
small_test_df = full_test_df.sample(n=1000, random_state=42)

# Tokenize
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
small_encodings = tokenizer(
    small_test_df["text"].tolist(),
    padding=True,
    truncation=True,
    return_tensors="pt"
)
small_labels = small_test_df["class"].tolist()
small_dataset = TextClassificationDataset(small_encodings, small_labels)

# Load trained model
model = DistilBertForSequenceClassification.from_pretrained("results/checkpoint-500")
model.to(device).eval()

# Inference
# Creates a batch loader for the small dataset
loader = DataLoader(small_dataset, batch_size=64, shuffle=False)

# Storage for predictions and labels
all_preds = []
all_labels = []

# Disables gradient calculations for inference
with torch.no_grad():
    # TQDM progress bar for the evaluation loop
    for batch in tqdm(loader, desc="Quick eval (1k samples)"):
        # Encodings and labels to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass
        logits = model(input_ids, attention_mask=attention_mask).logits
        preds = logits.argmax(dim=1)

        # Store predictions and labels
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Compute
accuracy  = accuracy_score(all_labels, all_preds)
precision, recall, f1, _ = precision_recall_fscore_support( # Don't include support
    all_labels, all_preds, average="binary"
)

# Print results
print("\nQuick Evaluation on 1000 sample Test Subset:")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
