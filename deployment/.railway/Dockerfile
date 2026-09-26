FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install minimal system dependencies (only what we need for download/extraction)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    unzip \
    && pip install --no-cache-dir gdown \
    && apt-get purge -y --auto-remove \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy deployment folder
COPY deployment/ /app/

# Which training run to serve. The Drive zip must contain
# backend/models/${MODEL_RUN}/final_model/, and research/results/${MODEL_RUN}/results.json
# must be committed: the API reports that file's held-out test metrics.
ARG MODEL_RUN=roberta-seed42
ENV MODEL_PATH=/app/backend/models/${MODEL_RUN}/final_model
COPY research/results/${MODEL_RUN}/results.json /app/backend/models/${MODEL_RUN}/results.json

# Download model from Google Drive at build time using gdown (handles large files)
ARG GDRIVE_FILE_ID=REPLACE_WITH_YOUR_FILE_ID
RUN if [ "$GDRIVE_FILE_ID" != "REPLACE_WITH_YOUR_FILE_ID" ]; then \
        echo "Downloading model from Google Drive (this may take 2-3 minutes)..."; \
        gdown "https://drive.google.com/uc?id=${GDRIVE_FILE_ID}" -O /tmp/model.zip && \
        echo "✓ Model downloaded successfully"; \
        echo "Extracting model..."; \
        unzip -qo /tmp/model.zip -d /app && \
        rm /tmp/model.zip && \
        MODEL_SIZE=$(du -h ${MODEL_PATH}/model.safetensors | cut -f1) && \
        echo "✓ Model extracted: $MODEL_SIZE"; \
    else \
        echo "⚠️  WARNING: Using embedded model (may be LFS pointer)"; \
        if [ -f "${MODEL_PATH}/model.safetensors" ]; then \
            MODEL_SIZE=$(du -h ${MODEL_PATH}/model.safetensors | cut -f1) && \
            echo "Model file size: $MODEL_SIZE"; \
        fi; \
    fi

# Verify model file exists and is valid size
RUN if [ ! -f "${MODEL_PATH}/model.safetensors" ]; then \
        echo "ERROR: Model file not found!"; \
        exit 1; \
    fi && \
    FILE_SIZE=$(stat -f%z ${MODEL_PATH}/model.safetensors 2>/dev/null || stat -c%s ${MODEL_PATH}/model.safetensors) && \
    if [ "$FILE_SIZE" -lt 1000000 ]; then \
        echo "ERROR: Model file is too small ($FILE_SIZE bytes). This is likely a Git LFS pointer file."; \
        echo "Please set GDRIVE_FILE_ID build argument or upload model to Google Drive."; \
        exit 1; \
    fi && \
    MODEL_SIZE=$(du -h ${MODEL_PATH}/model.safetensors | cut -f1) && \
    echo "✓ Model verified: $MODEL_SIZE"

# Install Python dependencies
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Save the base tokenizer next to the model so startup doesn't depend on the Hugging Face Hub
RUN python -c "import json; from transformers import AutoTokenizer; \
base = {'roberta': 'roberta-base', 'bert': 'bert-base-uncased', 'distilbert': 'distilbert-base-uncased', 'electra': 'google/electra-base-discriminator'}; \
model_type = json.load(open('${MODEL_PATH}/config.json'))['model_type']; \
AutoTokenizer.from_pretrained(base[model_type]).save_pretrained('${MODEL_PATH}')" && \
    echo "✓ Tokenizer saved to ${MODEL_PATH}"

# Change to backend directory for runtime
WORKDIR /app/backend

EXPOSE 8000

# Health check - uses PORT env var from Railway
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Use shell form to expand PORT environment variable
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}
