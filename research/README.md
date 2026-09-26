# Research Pipeline

This directory contains the model-training and evaluation code for the text-classification portion of the project.

## Results

The benchmark (DistilBERT, BERT, ELECTRA, RoBERTa, three seeds each, plus a TF-IDF baseline) and an audit of what the test score means are in the root README under "What the 99.45% does and doesn't mean". Per-run output is in `results/<model>-seed<seed>/results.json`; the audit output is `results/audit/audit.json`, produced by `scripts/audit_evaluation.py`.

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
  --base_config config/base_config.yaml \
  --model_config config/model_configs/distilbert.yaml \
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
