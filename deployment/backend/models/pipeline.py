"""
Analysis pipeline for suicide ideation detection.

Combines PII redaction, ML model prediction, and risk assessment.
"""

import logging
from typing import Dict

from services.pii import anonymize
from services.risk import score as keyword_score
from models.model_loader import model_loader
from models.interpretability import explain_prediction

logger = logging.getLogger(__name__)


class AnalysisPipeline:
    """Main analysis pipeline combining all components."""

    def __init__(self) -> None:
        self.use_ml_model = model_loader.is_available
        if self.use_ml_model:
            logger.info("✓ ML model available - using transformer-based predictions")
        else:
            logger.warning("⚠️ ML model not available - using keyword-based fallback")

    def analyze(self, text: str, explain: bool = False) -> Dict:
        """
        Analyze text for suicide ideation risk.

        Args:
            text: Raw text to analyze
            explain: Whether to include interpretability (text highlighting)
                    NOTE: Integrated Gradients is CPU-intensive. Default is False for performance.

        Returns:
            Dict with risk assessment, PII info, and optional explanations
        """
        # Step 1: PII Redaction
        redacted_text, pii_counts = anonymize(text)

        # Step 2: Risk Assessment
        if self.use_ml_model and model_loader.is_available:
            risk_result = self._ml_risk_assessment(redacted_text, explain)
        else:
            risk_result = self._keyword_risk_assessment(redacted_text)

        # Step 3: Format response
        # Do not return raw text. Provide optional snippet for context.
        snippet = redacted_text[:240] + ("..." if len(redacted_text) > 240 else "")

        return {
            "pii_redacted": pii_counts,
            "risk": risk_result,
            "sanitized_excerpt": snippet,
            "method": "ml_model" if self.use_ml_model else "keyword_fallback"
        }

    def _ml_risk_assessment(self, text: str, explain: bool) -> Dict:
        """
        Risk assessment using trained ML model.

        Args:
            text: Redacted text
            explain: Whether to include explanations

        Returns:
            Dict with risk_level, score, guidance, and optional highlighting
        """
        try:
            if explain:
                # Get full explanation with text highlighting
                explanation = explain_prediction(
                    model=model_loader.model,
                    tokenizer=model_loader.tokenizer,
                    text=text,
                    device=str(model_loader.device),
                    method='integrated_gradients',
                    max_length=256
                )

                predicted_class = explanation['predicted_class']
                confidence = explanation['confidence']
                highlighted_words = explanation['highlighted_words']

                # Extract most important phrases for triggers
                important_words = [
                    w['text'] for w in highlighted_words
                    if w['is_important'] and predicted_class == 1
                ][:10]  # Top 10

                # Extract problematic sentences
                problematic_sentences = self._extract_problematic_sentences(
                    text, highlighted_words, predicted_class
                )

            else:
                # Simple prediction without explanation (faster)
                prediction = model_loader.predict(text)
                if not prediction:
                    return self._keyword_risk_assessment(text)

                predicted_class = prediction['predicted_class']
                confidence = prediction['confidence']
                highlighted_words = None
                important_words = []

            # Map to risk levels
            if predicted_class == 1:  # Suicide
                if confidence >= 0.90:
                    risk_level = "HIGH"
                    guidance = "High-risk indicators detected. Alert appropriate staff immediately and follow your school's escalation protocol."
                elif confidence >= 0.70:
                    risk_level = "MEDIUM"
                    guidance = "Concerning signals present. Review with care and consider a check-in with the student per policy."
                else:
                    risk_level = "MEDIUM"
                    guidance = "Some concerning language detected. Review in context and follow institutional guidelines."
            else:  # Non-suicide
                risk_level = "LOW"
                guidance = "No strong indicators of suicidal ideation detected. Still review in context."

            result = {
                "risk_level": risk_level,
                "score": confidence if predicted_class == 1 else (1 - confidence),
                "confidence": confidence,
                "prediction": "suicide" if predicted_class == 1 else "non-suicide",
                "triggers": important_words,
                "guidance": guidance,
                "needs_human_review": True,
            }

            # Add highlighting if requested
            if explain and highlighted_words:
                result["highlighted_words"] = highlighted_words
                result["problematic_sentences"] = problematic_sentences

            return result

        except Exception as e:
            logger.error(f"ML risk assessment failed: {e}")
            logger.info("Falling back to keyword-based assessment")
            return self._keyword_risk_assessment(text)

    def _keyword_risk_assessment(self, text: str) -> Dict:
        """
        Fallback risk assessment using keyword matching.

        Args:
            text: Redacted text

        Returns:
            Dict with risk_level, score, triggers, and guidance
        """
        return keyword_score(text)

    def _extract_problematic_sentences(self, text: str, highlighted_words: list, predicted_class: int) -> list:
        """
        Extract sentences that contain high-importance words.

        Args:
            text: Original text
            highlighted_words: List of word importance scores
            predicted_class: 0 (non-suicide) or 1 (suicide)

        Returns:
            List of sentences with their average importance scores
        """
        import re

        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences or predicted_class == 0:
            return []

        # Create word to score mapping
        word_scores = {}
        word_index = 0
        for word_info in highlighted_words:
            word_scores[word_index] = word_info.get('normalized_score', 0)
            word_index += 1

        # Calculate sentence scores
        sentence_scores = []
        word_idx = 0

        for sentence in sentences:
            # Count words in this sentence (approximate by splitting on spaces)
            words_in_sentence = sentence.split()
            num_words = len(words_in_sentence)

            # Calculate average importance for this sentence
            sentence_score = 0
            count = 0
            for _ in range(num_words):
                if word_idx < len(highlighted_words):
                    sentence_score += word_scores.get(word_idx, 0)
                    count += 1
                    word_idx += 1

            avg_score = sentence_score / count if count > 0 else 0

            # Only include sentences with high average importance
            if avg_score >= 0.5:  # Threshold for "problematic"
                sentence_scores.append({
                    'sentence': sentence,
                    'score': avg_score
                })

        # Sort by score and return top 3
        sentence_scores.sort(key=lambda x: x['score'], reverse=True)
        return sentence_scores[:3]
