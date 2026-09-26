"""
Download model from Google Drive on Railway startup.
Run this before starting the server if model doesn't exist locally.
"""
import os
import urllib.request
import zipfile
import sys

MODEL_DIR = "/app/backend/models/roberta-seed42/final_model"
GDRIVE_FILE_ID = "1ialJXEYmk6Hhhq_nYQUqLTmIp-v5f5g9"  # roberta-model.zip, same file Railway builds from
MODEL_ZIP_URL = f"https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}"

def download_model():
    """Download and extract model if not present."""
    if os.path.exists(MODEL_DIR):
        print(f"✓ Model already exists at {MODEL_DIR}")
        return True

    print("⬇️  Downloading model from Google Drive...")
    try:
        # Create directories
        os.makedirs("/app", exist_ok=True)

        # Download zip
        zip_path = "/tmp/roberta-model.zip"
        urllib.request.urlretrieve(MODEL_ZIP_URL, zip_path)
        print(f"✓ Downloaded model to {zip_path}")

        # Extract
        print("📦 Extracting model...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # The zip contains backend/models/roberta-seed42/final_model/
            zip_ref.extractall("/app")

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
