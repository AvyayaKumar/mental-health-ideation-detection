# Research Pipeline

This directory contains the model-training and evaluation code for the text-classification portion of the project.

## Current experiment

The completed DistilBERT run is stored under `results/distilbert-seed42/`.

Test-set results:

| Metric | Value |
| --- | ---: |
| Accuracy | 0.9796 |
| Precision | 0.9824 |
| Recall | 0.9766 |
| F1 | 0.9795 |
| False-negative rate | 0.0234 |
| False-positive rate | 0.0175 |

Training configuration:

- model: `distilbert-base-uncased`
- seed: 42
- learning rate: `2e-5`
- batch size: 16
- epochs: 3
- max sequence length: 256

The full run output is in `results/distilbert-seed42/results.json`.

## Layout

```text
research/
├── config/
│   ├── base_config.yaml
│   └── model_configs/
├── data/
│   └── splits/
├── results/
├── scripts/
├── src/
│   ├── dataset.py
│   ├── metrics.py
│   ├── model.py
│   ├── train.py
│   └── utils.py
└── requirements-research.txt
```

## Training

Install dependencies:

```bash
pip install -r requirements-research.txt
```

Run a configured transformer experiment:

```bash
python src/train.py \
  --base-config config/base_config.yaml \
  --model-config config/model_configs/distilbert.yaml \
  --seed 42
```

Model-specific YAML files allow experiments to share the same data and training configuration while changing architecture-level settings.

## Baseline

A TF-IDF + logistic-regression baseline is available at:

```text
scripts/train_baseline.py
```

It uses the same saved train/validation/test split so its results can be compared against the transformer runs.

## Evaluation

The evaluation code records:

- accuracy
- precision
- recall
- F1
- false-negative rate
- false-positive rate
- confusion-matrix counts

Experiment metadata and results are written to JSON for later comparison.

## Reproducibility

The pipeline uses fixed split indices and explicit random seeds. Model and training settings are stored in YAML, and each completed run saves its configuration and metrics alongside the model output.

For the broader study design and planned comparisons, see `../docs/RESEARCH_PLAN.md`.
