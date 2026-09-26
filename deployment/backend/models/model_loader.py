"""
Model loader for suicide ideation detection.

Loads the trained transformer model and provides prediction interface.
"""

import os
import json
import logging
from typing import Optional, Dict
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)


BASE_TOKENIZERS = {
    "roberta": "roberta-base",
    "bert": "bert-base-uncased",
    "distilbert": "distilbert-base-uncased",
    "electra": "google/electra-base-discriminator",
}

MODEL_DISPLAY_NAMES = {
    "roberta": "RoBERTa",
    "bert": "BERT",
    "distilbert": "DistilBERT",
    "electra": "ELECTRA",
}


class ModelLoader:
    """Singleton model loader to avoid reloading model for each request."""

    _instance = None
    _model = None
    _tokenizer = None
    _device = None
    _info = None

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

            self._tokenizer = self._load_tokenizer(model_path)

            self._model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self._model.to(self._device)
            self._model.eval()

            self._info = self._load_info(model_path)
            logger.info(f"✓ Model loaded successfully on {self._device}: {self._info}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self._model = None
            self._tokenizer = None

    @staticmethod
    def _load_tokenizer(model_path: str):
        """
        Use the tokenizer saved next to the model if there is one; otherwise use
        the base model's tokenizer. Only look in model_path when tokenizer files
        exist: with no vocab there, newer transformers releases build an empty
        tokenizer instead of raising, and every input collapses to the same two
        special tokens.
        """
        tokenizer_files = ("tokenizer.json", "vocab.json", "vocab.txt")
        if any(os.path.exists(os.path.join(model_path, f)) for f in tokenizer_files):
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        else:
            with open(os.path.join(model_path, "config.json")) as f:
                model_type = json.load(f)["model_type"]
            base = BASE_TOKENIZERS[model_type]
            logger.info(f"No tokenizer files in {model_path}; using {base}")
            tokenizer = AutoTokenizer.from_pretrained(base)
        if len(tokenizer) < 1000:
            raise RuntimeError(f"Tokenizer has only {len(tokenizer)} tokens; refusing to serve predictions")
        return tokenizer

    @staticmethod
    def _load_info(model_path: str) -> Dict:
        """Model name from the loaded config, and test metrics from that run's results.json."""
        with open(os.path.join(model_path, "config.json")) as f:
            model_type = json.load(f)["model_type"]
        info = {"name": MODEL_DISPLAY_NAMES.get(model_type, model_type), "model_type": model_type}

        results_path = os.getenv("MODEL_RESULTS_PATH") or os.path.join(os.path.dirname(model_path), "results.json")
        try:
            with open(results_path) as f:
                test = json.load(f)["test_results"]
            info["test_accuracy"] = test["eval_accuracy"]
            info["test_f1"] = test["eval_f1"]
            info["test_fnr"] = test["eval_fnr"]
        except (OSError, KeyError, ValueError) as e:
            logger.warning(f"No test metrics for {model_path} ({results_path}): {e}")
        return info

    @property
    def info(self) -> Optional[Dict]:
        """Name and held-out test metrics of the loaded model, or None if no model is loaded."""
        return self._info if self.is_available else None

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
