"""
Model loader for suicide ideation detection.

Loads the trained transformer model and provides prediction interface.
"""

import os
import logging
from typing import Optional, Dict
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)


class ModelLoader:
    """Singleton model loader to avoid reloading model for each request."""

    _instance = None
    _model = None
    _tokenizer = None
    _device = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._model is None:
            self._load_model()

    def _load_model(self):
        """Load the model and tokenizer."""
        model_path = os.getenv("MODEL_PATH", None)

        if not model_path:
            logger.warning("MODEL_PATH not set. Model predictions will not be available.")
            logger.warning("Set MODEL_PATH environment variable to your trained model directory.")
            self._model = None
            self._tokenizer = None
            return

        if not os.path.exists(model_path):
            logger.warning(f"Model path does not exist: {model_path}")
            logger.warning("Model predictions will not be available until model is trained.")
            self._model = None
            self._tokenizer = None
            return

        try:
            logger.info(f"Loading model from: {model_path}")
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Load tokenizer from base model (fallback if not in model_path)
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(model_path)
            except:
                logger.info("Tokenizer not found in model directory, loading from base model")
                # Detect model type from path and use appropriate base tokenizer
                if "roberta" in model_path.lower():
                    self._tokenizer = AutoTokenizer.from_pretrained("roberta-base")
                elif "bert" in model_path.lower() and "distil" not in model_path.lower():
                    self._tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
                else:
                    self._tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

            self._model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self._model.to(self._device)
            self._model.eval()

            logger.info(f"✓ Model loaded successfully on {self._device}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self._model = None
            self._tokenizer = None

    @property
    def is_available(self) -> bool:
        """Check if model is loaded and available."""
        return self._model is not None and self._tokenizer is not None

    @property
    def model(self):
        """Get the loaded model."""
        return self._model

    @property
    def tokenizer(self):
        """Get the loaded tokenizer."""
        return self._tokenizer

    @property
    def device(self):
        """Get the device (cpu/cuda)."""
        return self._device

    def predict(self, text: str, max_length: int = 256) -> Optional[Dict]:
        """
        Make a simple prediction without explanation.

        Args:
            text: Input text to analyze
            max_length: Maximum sequence length

        Returns:
            Dict with prediction, confidence, and class, or None if model not available
        """
        if not self.is_available:
            return None

        try:
            # Tokenize
            inputs = self._tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            # Predict
            with torch.no_grad():
                outputs = self._model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                pred = torch.argmax(probs, dim=1)

            prediction = int(pred[0])
            confidence = float(probs[0][prediction])

            return {
                'prediction': 'suicide' if prediction == 1 else 'non-suicide',
                'predicted_class': prediction,
                'confidence': confidence
            }

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return None


# Global instance
model_loader = ModelLoader()
