"""
PART 1: Load and Explore the Dataset
"""

import pandas as pd

# TODO: Replace with your actual CSV file path (e.g., "data.csv")
csv_path = "Suicide_Detection.csv"

# Step 1: Load the dataset
# Fill in below
df = pd.DataFrame(csv_path)

# Step 2: Print dataset info
# Fill in below
# print("Number of rows:", ...)
# print("Number of unique labels:", ...)

# Step 3: Print the dataset head
# print("\nDataset Head:\n", ...)


"""
PART 2: Tokenize the Text with DistilBERT (Just run this code)
"""

from transformers import DistilBertTokenizerFast

# Step 1: Load the tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# Step 2: Get a few sample texts from the dataset
sample_texts = df['text'].sample(n=4, random_state=1).tolist()

# Step 3: Tokenize the sample texts
encodings = tokenizer(sample_texts, padding=True, truncation=True, return_tensors="pt")

# Step 4: Print input IDs and attention masks
print("\nTokenized Input IDs:\n", encodings['input_ids'])
print("\nAttention Masks:\n", encodings['attention_mask'])

# Step 5: Decode each input ID back to text
print("\nDecoded Inputs:")
for ids in encodings['input_ids']:
    print(" -", tokenizer.decode(ids))
