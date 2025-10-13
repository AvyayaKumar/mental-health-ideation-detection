#!/usr/bin/env python3
"""
Upload model to Hugging Face Hub for Railway deployment
Run this once: python scripts/upload_model_to_hf.py
"""
from huggingface_hub import HfApi, create_repo
from pathlib import Path
import os

# Configuration
MODEL_PATH = "results/distilbert-seed42/final_model"
HF_USERNAME = "YOUR_HF_USERNAME"  # Change this!
REPO_NAME = "suicide-ideation-detection-distilbert"

def upload_model():
    """Upload model to Hugging Face Hub"""
    api = HfApi()

    # Create repository
    repo_id = f"{HF_USERNAME}/{REPO_NAME}"
    print(f"Creating repository: {repo_id}")

    try:
        create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=True)
        print(f"✓ Repository created (or already exists)")
    except Exception as e:
        print(f"Note: {e}")

    # Upload model files
    print(f"\nUploading model files from {MODEL_PATH}...")
    api.upload_folder(
        folder_path=MODEL_PATH,
        repo_id=repo_id,
        repo_type="model",
    )

    print(f"\n✅ Model uploaded successfully!")
    print(f"   Model ID: {repo_id}")
    print(f"\n📝 Add this to your Railway environment variables:")
    print(f"   HF_MODEL_ID={repo_id}")
    print(f"\n🔑 If private, also add your HF token:")
    print(f"   HF_TOKEN=your_token_here")
    print(f"   Get it from: https://huggingface.co/settings/tokens")

if __name__ == "__main__":
    # Check if logged in
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("⚠️  Please set HF_TOKEN environment variable")
        print("   Get your token from: https://huggingface.co/settings/tokens")
        print("\n   Then run:")
        print("   export HF_TOKEN=your_token_here")
        print("   python scripts/upload_model_to_hf.py")
        exit(1)

    if HF_USERNAME == "YOUR_HF_USERNAME":
        print("⚠️  Please edit this script and set your HF_USERNAME")
        exit(1)

    upload_model()
