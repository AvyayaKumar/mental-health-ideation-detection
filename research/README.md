# Research - Transformer Model Benchmarking for Suicide Ideation Detection

## 🎯 Research Objective

Conduct a comprehensive benchmark of transformer models for suicide ideation detection with rigorous experimental methodology suitable for peer-reviewed publication.

## 📋 Research Plan

### Phase 1: Setup & Infrastructure (Week 1-2)
- ✅ Data preprocessing & EDA
- ✅ Create master training script
- ✅ Setup experiment tracking (wandb)
- ✅ Implement evaluation metrics

### Phase 2: Model Training (Week 3-7)
Models to train (priority order):
1. ✅ DistilBERT (baseline complete)
2. 🔄 BERT-base
3. ⏳ RoBERTa-base
4. ⏳ ELECTRA-base
5. ⏳ DeBERTa-v3-base
6. ⏳ XLNet-base

**Training Protocol**:
- 3 random seeds per model (42, 123, 456)
- Standard hyperparameters: LR=2e-5, batch=16, epochs=3
- Max sequence length: 128
- Report mean ± std for all metrics

### Phase 3: Hyperparameter Tuning (Week 7-9)
- Learning rates: [1e-5, 2e-5, 5e-5]
- Batch sizes: [16, 32]
- Sequence lengths: [128, 256]
- Class weights vs focal loss

### Phase 4: Error Analysis (Week 9-10)
- Confusion matrix analysis
- False Negative deep dive
- Linguistic pattern identification
- SHAP/LIME interpretability

### Phase 5: Statistical Validation (Week 11-12)
- McNemar's test
- Significance testing
- Generate publication-ready figures

## 🏃 Quick Start

### Installation

```bash
cd research/
pip install -r requirements-research.txt
```

### Training a Model

```bash
# Train DistilBERT with default config
python src/train.py --config config/distilbert_base.yaml

# Train with custom seed
python src/train.py --config config/distilbert_base.yaml --seed 123

# Train BERT
python src/train.py --config config/bert_base.yaml
```

### Evaluation

```bash
# Evaluate a trained model
python scripts/evaluate.py \
  --model_path results/distilbert-seed42/final_model \
  --test_data data/processed/test.csv

# Generate SHAP analysis
python scripts/interpret.py \
  --model_path results/distilbert-seed42/final_model \
  --samples 100
```

## 📊 Evaluation Metrics

**Primary Metrics**:
1. **False Negative Rate (FNR)** - Most critical for safety
2. **F1 Score** - Primary performance metric
3. **Recall** - Sensitivity to positive cases
4. **Precision** - Accuracy of positive predictions
5. **Accuracy** - Overall correctness

**Additional Metrics**:
- Confusion matrix
- Precision-Recall curve
- ROC-AUC
- Per-class metrics

## 🗂️ Directory Structure

```
research/
├── src/                       # Source code
│   ├── train.py              # Main training script
│   ├── dataset.py            # Data loading & preprocessing
│   ├── model.py              # Model definitions
│   ├── metrics.py            # Evaluation metrics
│   ├── interpretability.py   # SHAP/LIME
│   └── utils.py              # Utilities
│
├── config/                    # Experiment configurations
│   ├── distilbert_base.yaml
│   ├── bert_base.yaml
│   ├── roberta_base.yaml
│   └── experiment_configs/
│
├── scripts/                   # Training & analysis scripts
│   ├── train_baseline.py
│   ├── evaluate.py
│   └── interpret.py
│
├── results/                   # Model checkpoints & metrics
│   └── distilbert-seed42/
│       ├── final_model/      # Trained model
│       ├── checkpoints/      # Training checkpoints
│       └── results.json      # Metrics
│
├── data/                      # Datasets (NOT in git)
│   ├── raw/                  # Original CSV
│   ├── processed/            # Train/val/test splits
│   └── cache/                # Tokenized cache
│
└── notebooks/                 # Jupyter notebooks (if any)
```

## 📈 Experiment Tracking

We use **Weights & Biases (wandb)** for experiment tracking.

```bash
# Login to wandb
wandb login

# Training automatically logs to wandb
python src/train.py --config config/distilbert_base.yaml
```

View experiments at: https://wandb.ai/your-username/suicide-ideation-detection

## 🔬 Running Experiments

### Baseline Experiment (DistilBERT)

```bash
# Already completed - see results/distilbert-seed42/
python src/train.py --config config/distilbert_base.yaml --seed 42
python src/train.py --config config/distilbert_base.yaml --seed 123
python src/train.py --config config/distilbert_base.yaml --seed 456
```

### Additional Models

```bash
# BERT
python src/train.py --config config/bert_base.yaml --seed 42

# RoBERTa
python src/train.py --config config/roberta_base.yaml --seed 42

# ELECTRA
python src/train.py --config config/electra_base.yaml --seed 42
```

### Hyperparameter Tuning

```bash
# Grid search example
for lr in 1e-5 2e-5 5e-5; do
  for bs in 16 32; do
    python src/train.py \
      --config config/distilbert_base.yaml \
      --learning_rate $lr \
      --batch_size $bs \
      --seed 42
  done
done
```

## 📊 Results Summary

### Current Results (DistilBERT - Seed 42)

| Metric | Value |
|--------|-------|
| Accuracy | TBD |
| F1 Score | TBD |
| Recall | TBD |
| Precision | TBD |
| FNR | TBD |

*Run evaluation script to populate results*

## 🐛 Troubleshooting

**CUDA Out of Memory**:
- Reduce batch size: `--batch_size 8`
- Use gradient accumulation: `--gradient_accumulation_steps 2`
- Reduce max sequence length: `--max_length 128`

**Dataset Not Found**:
- Ensure CSV is in `data/raw/`
- Run preprocessing: `python src/dataset.py --preprocess`

**Model Loading Errors**:
- Check model path exists
- Verify all model files present (config.json, model.safetensors, tokenizer files)

## 📝 Publication Checklist

- [ ] Complete all model training (6 models × 3 seeds)
- [ ] Hyperparameter tuning for top 3 models
- [ ] Error analysis & confusion matrices
- [ ] SHAP/LIME interpretability analysis
- [ ] Statistical significance testing
- [ ] Generate all publication figures
- [ ] Write methodology section
- [ ] Write results section
- [ ] Write discussion section
- [ ] Prepare supplementary materials

## 📚 Key References

1. Transformer Models: Vaswani et al., "Attention Is All You Need"
2. BERT: Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers"
3. Mental Health NLP: [Add relevant papers]

## 🔐 Data Privacy

- **No real student data in repository**
- Dataset contains public suicide detection data
- All experiments comply with ethics guidelines
- Model designed for teacher assistance, not surveillance

---

For deployment of trained models, see [../deployment/README.md](../deployment/README.md)
