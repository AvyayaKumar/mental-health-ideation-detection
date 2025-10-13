"""
Download model from Google Drive on Railway startup.
Run this before starting the server if model doesn't exist locally.
"""
import os
import urllib.request
import zipfile
import sys

MODEL_DIR = "/app/models/distilbert-seed42/final_model"
GDRIVE_FILE_ID = "1rVmEb6WqLzNIdVbJOAcJnoN6xiM6NPlA"
MODEL_ZIP_URL = f"https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}"

def download_model():
    """Download and extract model if not present."""
    if os.path.exists(MODEL_DIR):
        print(f"✓ Model already exists at {MODEL_DIR}")
        return True

    print("⬇️  Downloading model from Google Drive...")
    try:
        # Create directories
        os.makedirs("/app/models", exist_ok=True)

        # Download zip
        zip_path = "/tmp/distilbert-model.zip"
        urllib.request.urlretrieve(MODEL_ZIP_URL, zip_path)
        print(f"✓ Downloaded model to {zip_path}")

        # Extract
        print("📦 Extracting model...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("/app/models")

        # Cleanup
        os.remove(zip_path)
        print(f"✓ Model extracted to {MODEL_DIR}")
        return True

    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        return False

if __name__ == "__main__":
    success = download_model()
    sys.exit(0 if success else 1)
