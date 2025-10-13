"""
Test script for interpretability features.

This script demonstrates how to:
1. Load a trained model
2. Generate explanations for predictions
3. Visualize important words/phrases
4. Compare different explanation methods
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from src.interpretability import explain_prediction, format_for_web_display


def main():
    """Test interpretability on example texts."""

    print("=" * 80)
    print("INTERPRETABILITY TEST")
    print("=" * 80)

    # Configuration
    MODEL_PATH = os.environ.get('MODEL_PATH', 'results/distilbert-seed42/final_model')
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    print(f"\nLoading model from: {MODEL_PATH}")
    print(f"Device: {DEVICE}")

    # Load model
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        model.to(DEVICE).eval()
        print("✓ Model loaded successfully\n")
    except Exception as e:
        print(f"⚠️  Could not load model: {e}")
        print("Using pretrained DistilBERT instead (won't give accurate results)\n")
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=2
        )
        model.to(DEVICE).eval()

    # Example texts
    examples = [
        {
            "label": "Non-suicidal",
            "text": "I'm really excited about my upcoming college applications. "
                    "I've been working hard on my essays and I think they're coming together nicely. "
                    "My family has been so supportive throughout this process."
        },
        {
            "label": "Potentially concerning",
            "text": "I don't see the point anymore. Nothing matters and I feel so alone. "
                    "I wish I could just disappear and not have to deal with any of this. "
                    "Nobody would even notice if I was gone."
        },
    ]

    # Test both methods
    methods = ["integrated_gradients", "attention"]

    for i, example in enumerate(examples, 1):
        print("=" * 80)
        print(f"EXAMPLE {i}: {example['label']}")
        print("=" * 80)
        print(f"\nText: {example['text']}\n")

        for method in methods:
            print(f"\n--- Method: {method.upper().replace('_', ' ')} ---\n")

            # Get explanation
            explanation = explain_prediction(
                model=model,
                tokenizer=tokenizer,
                text=example['text'],
                device=str(DEVICE),
                method=method,
                max_length=256
            )

            # Print results
            print(f"Prediction: {explanation['prediction']}")
            print(f"Confidence: {explanation['confidence']:.4f}")

            # Show top 10 most important words
            highlighted_words = explanation['highlighted_words']
            sorted_words = sorted(
                highlighted_words,
                key=lambda x: x['normalized_score'],
                reverse=True
            )[:10]

            print(f"\nTop 10 most important words:")
            for j, word_info in enumerate(sorted_words, 1):
                word = word_info['text']
                score = word_info['normalized_score']
                importance = "🔴" if word_info['is_important'] else "⚪"
                print(f"  {j:2d}. {importance} {word:20s} (score: {score:.3f})")

            # Show HTML output
            print(f"\nHTML output (for web display):")
            html = format_for_web_display(explanation)
            print(html[:200] + "..." if len(html) > 200 else html)

        print()

    print("=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    print("\nInterpretability features are working!")
    print("You can now use these in the web app to highlight important text.")
    print("\nTo start the web app:")
    print("  cd apps/")
    print("  MODEL_PATH=../results/distilbert-seed42/final_model python app.py")
    print("\nThen visit: http://127.0.0.1:8080")
    print()


if __name__ == "__main__":
    main()
