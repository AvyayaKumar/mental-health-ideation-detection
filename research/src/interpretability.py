"""
Interpretability utilities for suicide ideation detection models.

Provides methods to explain model predictions by highlighting
important words, phrases, and sentences in the input text.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def get_token_attributions(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    text: str,
    predicted_class: int,
    device: str = "cpu",
    max_length: int = 256
) -> Tuple[List[str], List[float]]:
    """
    Get token-level attribution scores using Integrated Gradients.

    Args:
        model: Fine-tuned transformer model
        tokenizer: Corresponding tokenizer
        text: Input text to explain
        predicted_class: The predicted class (0 or 1)
        device: Device to run on ('cpu' or 'cuda')
        max_length: Maximum sequence length

    Returns:
        tokens: List of tokens
        attributions: List of attribution scores (higher = more important)
    """
    try:
        from captum.attr import LayerIntegratedGradients
    except ImportError:
        raise ImportError("Please install captum: pip install captum")

    model.eval()
    model.zero_grad()

    # Tokenize input
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding="max_length"
    )

    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # Get baseline (all padding tokens)
    baseline_ids = torch.zeros_like(input_ids)
    baseline_ids[:, 0] = tokenizer.cls_token_id
    baseline_ids[:, 1] = tokenizer.sep_token_id

    # Define forward function for Integrated Gradients
    def predict(input_ids):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits[:, predicted_class]

    # Compute attributions using Integrated Gradients
    lig = LayerIntegratedGradients(predict, model.get_input_embeddings())

    attributions_ig = lig.attribute(
        inputs=input_ids,
        baselines=baseline_ids,
        n_steps=50,
        return_convergence_delta=False
    )

    # Sum attributions across embedding dimension
    attributions = attributions_ig.sum(dim=-1).squeeze(0)
    attributions = attributions.detach().cpu().numpy()

    # Get tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    # Filter out padding tokens
    valid_indices = attention_mask[0].cpu().numpy() == 1
    tokens = [t for t, v in zip(tokens, valid_indices) if v]
    attributions = attributions[valid_indices]

    return tokens, attributions.tolist()


def get_attention_scores(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    text: str,
    device: str = "cpu",
    max_length: int = 256
) -> Tuple[List[str], List[float]]:
    """
    Get attention-based importance scores (faster but less accurate).

    Args:
        model: Fine-tuned transformer model
        tokenizer: Corresponding tokenizer
        text: Input text to explain
        device: Device to run on
        max_length: Maximum sequence length

    Returns:
        tokens: List of tokens
        scores: List of attention scores
    """
    model.eval()

    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding="max_length"
    )

    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # Get attention weights
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True
        )

    # Average attention across all heads and layers
    attentions = torch.stack(outputs.attentions)  # (layers, batch, heads, seq, seq)
    avg_attention = attentions.mean(dim=[0, 2])  # Average over layers and heads

    # Get attention to CLS token (represents overall importance)
    cls_attention = avg_attention[0, :, 0]  # Attention from all tokens to CLS

    scores = cls_attention.cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    # Filter padding
    valid_indices = attention_mask[0].cpu().numpy() == 1
    tokens = [t for t, v in zip(tokens, valid_indices) if v]
    scores = scores[valid_indices]

    return tokens, scores.tolist()


def aggregate_subword_tokens(
    tokens: List[str],
    scores: List[float],
    tokenizer: AutoTokenizer
) -> Tuple[List[str], List[float]]:
    """
    Aggregate subword tokens (e.g., "##ing") back into full words.

    Args:
        tokens: List of subword tokens
        scores: Attribution scores for each token
        tokenizer: Tokenizer used

    Returns:
        words: List of full words
        word_scores: Aggregated scores for each word
    """
    words = []
    word_scores = []
    current_word = ""
    current_scores = []

    for token, score in zip(tokens, scores):
        # Skip special tokens
        if token in [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token]:
            continue

        # Check if it's a subword token
        if token.startswith("##"):
            current_word += token[2:]
            current_scores.append(score)
        else:
            # Save previous word if exists
            if current_word:
                words.append(current_word)
                word_scores.append(np.mean(current_scores))

            # Start new word
            current_word = token
            current_scores = [score]

    # Don't forget the last word
    if current_word:
        words.append(current_word)
        word_scores.append(np.mean(current_scores))

    return words, word_scores


def highlight_important_phrases(
    text: str,
    words: List[str],
    scores: List[float],
    threshold_percentile: float = 75
) -> List[Dict[str, any]]:
    """
    Identify and highlight important phrases based on attribution scores.

    Args:
        text: Original input text
        words: List of words
        scores: Attribution scores
        threshold_percentile: Percentile threshold for "important" (default: top 25%)

    Returns:
        List of dicts with 'text', 'score', and 'is_important' keys
    """
    # Normalize scores to [0, 1]
    scores = np.array(scores)
    if scores.max() > scores.min():
        normalized_scores = (scores - scores.min()) / (scores.max() - scores.min())
    else:
        normalized_scores = np.ones_like(scores)

    # Determine threshold
    threshold = np.percentile(normalized_scores, threshold_percentile)

    # Create highlighted spans
    highlighted = []
    for word, score, norm_score in zip(words, scores, normalized_scores):
        highlighted.append({
            'text': word,
            'score': float(score),
            'normalized_score': float(norm_score),
            'is_important': bool(norm_score >= threshold)
        })

    return highlighted


def explain_prediction(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    text: str,
    device: str = "cpu",
    method: str = "integrated_gradients",
    max_length: int = 256
) -> Dict:
    """
    Complete explanation pipeline for a prediction.

    Args:
        model: Fine-tuned model
        tokenizer: Tokenizer
        text: Input text to explain
        device: Device to run on
        method: 'integrated_gradients' or 'attention'
        max_length: Maximum sequence length

    Returns:
        Dict containing prediction, confidence, and highlighted text
    """
    model.to(device)
    model.eval()

    # Get prediction
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding="max_length"
    )

    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0, predicted_class].item()

    # Get attributions
    if method == "integrated_gradients":
        tokens, scores = get_token_attributions(
            model, tokenizer, text, predicted_class, device, max_length
        )
    elif method == "attention":
        tokens, scores = get_attention_scores(
            model, tokenizer, text, device, max_length
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    # Aggregate subwords
    words, word_scores = aggregate_subword_tokens(tokens, scores, tokenizer)

    # Highlight important phrases
    highlighted = highlight_important_phrases(text, words, word_scores)

    return {
        'prediction': 'suicide' if predicted_class == 1 else 'non-suicide',
        'predicted_class': predicted_class,
        'confidence': confidence,
        'highlighted_words': highlighted,
        'method': method
    }


def format_for_web_display(explanation: Dict) -> str:
    """
    Format explanation as HTML for web display.

    Args:
        explanation: Output from explain_prediction()

    Returns:
        HTML string with highlighted text
    """
    html_parts = []

    for item in explanation['highlighted_words']:
        word = item['text']
        score = item['normalized_score']
        is_important = item['is_important']

        if is_important:
            # Color intensity based on score
            opacity = 0.3 + (score * 0.7)  # Range: 0.3 to 1.0
            color = f"rgba(255, 0, 0, {opacity})" if explanation['predicted_class'] == 1 else f"rgba(0, 128, 255, {opacity})"
            html_parts.append(
                f'<span style="background-color: {color}; padding: 2px 4px; border-radius: 3px;" '
                f'title="Score: {score:.3f}">{word}</span>'
            )
        else:
            html_parts.append(word)

        # Add space after word
        html_parts.append(' ')

    return ''.join(html_parts)
