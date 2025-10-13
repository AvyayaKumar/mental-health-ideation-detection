import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

# Point to the fine-tuned checkpoint
CKPT = "results/checkpoint-500"

# Load the pretrained tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
# Load weights from the local checkpoint
model = DistilBertForSequenceClassification.from_pretrained(CKPT)

# Move to GPU/CPU (performance)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device).eval()

# Batch‐inference
examples = [
    "I feel like hurting myself today.",
    "I'm so happy I got a promotion!"
]

# Tokenize the examples
enc = tokenizer(examples, padding=True, truncation=True, max_length=128, return_tensors="pt")
# Move to GPU/CPU
enc = {k: v.to(device) for k,v in enc.items()}

# Inference
with torch.no_grad(): # Disales gradient calculations (since we are not training)
    logits = model(**enc).logits # Forward pass (logtis are just the raw scores)
    probs  = torch.softmax(logits, dim=-1) # Convert logits to probabilities
    preds  = torch.argmax(probs, dim=1) # Get the predicted class

# Print the results and probabilities
for text, p, prob in zip(examples, preds.cpu(), probs.cpu().tolist()):
    print(f"{text!r} → label={int(p)} (score={prob[p]:.4f})")
