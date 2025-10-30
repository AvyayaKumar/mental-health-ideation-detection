FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
# Updated: 2025-10-12 - Fixed Google Drive file ID

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

# Download model from Google Drive at build time using gdown (handles large files)
ARG GDRIVE_FILE_ID=REPLACE_WITH_YOUR_FILE_ID
RUN if [ "$GDRIVE_FILE_ID" != "REPLACE_WITH_YOUR_FILE_ID" ]; then \
        echo "Downloading model from Google Drive (this may take 2-3 minutes)..."; \
        gdown "https://drive.google.com/uc?id=${GDRIVE_FILE_ID}" -O /tmp/model.zip && \
        echo "✓ Model downloaded successfully"; \
        echo "Extracting model..."; \
        unzip -qo /tmp/model.zip -d /app && \
        rm /tmp/model.zip && \
        MODEL_SIZE=$(du -h /app/backend/models/roberta-seed42/final_model/model.safetensors | cut -f1) && \
        echo "✓ Model extracted: $MODEL_SIZE"; \
    else \
        echo "⚠️  WARNING: Using embedded model (may be LFS pointer)"; \
        if [ -f "/app/backend/models/roberta-seed42/final_model/model.safetensors" ]; then \
            MODEL_SIZE=$(du -h /app/backend/models/roberta-seed42/final_model/model.safetensors | cut -f1) && \
            echo "Model file size: $MODEL_SIZE"; \
        fi; \
    fi

# Verify model file exists and is valid size
RUN if [ ! -f "/app/backend/models/roberta-seed42/final_model/model.safetensors" ]; then \
        echo "ERROR: Model file not found!"; \
        exit 1; \
    fi && \
    FILE_SIZE=$(stat -f%z /app/backend/models/roberta-seed42/final_model/model.safetensors 2>/dev/null || stat -c%s /app/backend/models/roberta-seed42/final_model/model.safetensors) && \
    if [ "$FILE_SIZE" -lt 1000000 ]; then \
        echo "ERROR: Model file is too small ($FILE_SIZE bytes). This is likely a Git LFS pointer file."; \
        echo "Please set GDRIVE_FILE_ID build argument or upload model to Google Drive."; \
        exit 1; \
    fi && \
    MODEL_SIZE=$(du -h /app/backend/models/roberta-seed42/final_model/model.safetensors | cut -f1) && \
    echo "✓ Model verified: $MODEL_SIZE"

# Install Python dependencies
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Change to backend directory for runtime
WORKDIR /app/backend

EXPOSE 8000

# Health check - uses PORT env var from Railway
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Use shell form to expand PORT environment variable
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}
