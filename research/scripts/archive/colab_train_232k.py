# Google Colab commands to run the 232K model training
# Copy and paste these commands into Colab cells

# Cell 1: Install dependencies
!pip install transformers accelerate torch safetensors scikit-learn pandas

# Cell 2: Upload the large dataset
from google.colab import files
print("Please upload 'Suicide_Detection 2.csv' file")
uploaded = files.upload()

# Rename to match script expectation
import os
if 'Suicide_Detection 2.csv' in uploaded:
    os.rename('Suicide_Detection 2.csv', 'Suicide_Detection_Full.csv')

# Cell 3: Upload the training script
print("Please upload 'model_training_230k.py' file")
uploaded = files.upload()

# Cell 4: Run the training
!python model_training_230k.py

# Cell 5: Zip and download results
!zip -r results_232k.zip results_230k
from google.colab import files
files.download('results_232k.zip')

# Cell 6: Optional - Evaluate the model
# First upload evaluate_model.py if you have it
# !python evaluate_model.py