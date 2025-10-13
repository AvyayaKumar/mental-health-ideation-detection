# Comprehensive Research Plan: Suicide Ideation Detection Using Transformer Models

## Executive Summary
A rigorous 7-12 week research project to benchmark transformer models for suicide ideation detection on a 232K sample dataset. The project employs best practices for reproducibility, statistical validation, and interpretability, suitable for academic publication.

**Platform:** Google Colab Pro (24hr sessions, GPU acceleration)
**Timeline:** 7-12 weeks (full-time equivalent)
**Expected Contribution:** Comprehensive benchmark + domain-specific insights from error analysis

---

## Research Objectives

1. **Primary:** Compare performance of 6 transformer architectures + traditional baselines for suicide ideation classification
2. **Secondary:** Identify linguistic patterns that lead to model failures through error analysis
3. **Tertiary:** Develop optimal ensemble methods for production deployment
4. **Impact:** Establish best practices for medical text classification with critical safety implications

---

## Model Architectures (Priority Order)

### Transformer Models
1. **DistilBERT** (66M params) - Current baseline, fastest inference
2. **BERT-base** (110M params) - Standard transformer baseline
3. **RoBERTa-base** (125M params) - Optimized pretraining approach
4. **ELECTRA-base** (110M params) - **Priority addition** - Sample-efficient, faster training
5. **XLNet-base** (110M params) - Permutation language modeling
6. **DeBERTa-v3-base** (86M params) - Disentangled attention mechanism

### Traditional Baselines (Essential for Publication)
7. **TF-IDF + Logistic Regression** - Classic strong baseline
8. **TF-IDF + SVM** - Alternative traditional approach
9. **Simple LSTM/GRU** - Demonstrate advantage of transformer architecture

---

## Evaluation Metrics (Priority Order)

Given the critical nature of suicide detection:
1. **False Negative Rate (FNR)** - Most critical (missing actual suicide ideation)
2. **F1 Score** - Primary metric for publication
3. **Recall** - Sensitivity to positive cases
4. **Precision** - Avoiding false alarms
5. **Accuracy** - Overall performance

**Reporting Standard:** Mean ± std across 3 random seeds (42, 123, 456)

---

## Phase-by-Phase Implementation Plan

### PHASE 0: Data Validation (Week 1, Days 1-2)
**Critical First Step - Do NOT skip**

#### Day 1: Exploratory Data Analysis
- [ ] Check class distribution in `Suicide_Detection_Full.csv`
- [ ] Analyze text length distribution (verify most samples < 256 tokens)
- [ ] Check for missing values, duplicates
- [ ] Document class imbalance ratio
- [ ] Create EDA notebook with visualizations

#### Day 2: Data Splitting & Validation
- [ ] Implement 80/10/10 stratified split (train/val/test)
- [ ] Use fixed random_state=42 for reproducibility
- [ ] **Save split indices** to JSON/pickle file
- [ ] **Verify split:** Check class distribution in each split
- [ ] Confirm stratification worked correctly
- [ ] Document: train_size, val_size, test_size with percentages

**Output:**
- `data_splits.json` (saved indices)
- `eda.ipynb` (exploratory analysis)
- Class distribution report

---

### PHASE 1: Infrastructure Setup (Week 1, Days 3-7)

#### Day 3-4: Project Scaffolding
```
project_structure/
├── config/
│   ├── base_config.yaml
│   ├── model_configs/
│   │   ├── distilbert.yaml
│   │   ├── bert.yaml
│   │   ├── roberta.yaml
│   │   └── ...
│   └── experiment_configs/
│       └── hyperparameter_grids.yaml
├── src/
│   ├── dataset.py          # Dataset class with caching
│   ├── model.py            # Unified model loading
│   ├── train.py            # Master training script
│   ├── metrics.py          # All evaluation metrics
│   ├── utils.py            # Helper functions
│   └── ensemble.py         # Ensemble methods
├── scripts/
│   ├── train_baseline.py   # TF-IDF baselines
│   └── run_experiments.sh  # Batch experiment runner
├── notebooks/
│   ├── eda.ipynb
│   ├── error_analysis.ipynb
│   └── results_visualization.ipynb
├── results/
│   ├── models/             # Saved checkpoints
│   ├── logs/               # Training logs
│   └── experiments.csv     # All experiment results
└── data/
    ├── raw/
    ├── processed/
    └── splits/             # Saved split indices
```

#### Day 4-5: Configuration System
- [ ] Create `base_config.yaml` with common parameters
- [ ] Individual model configs (model name, tokenizer, hyperparams)
- [ ] Implement config loading utility in `utils.py`
- [ ] Support for config inheritance/overrides

#### Day 5: Core Modules (First Pass)
**dataset.py**
- [ ] Load data from saved split indices
- [ ] Tokenization with caching (save to disk)
- [ ] Support for different max_seq_lengths [128, 256]
- [ ] PyTorch Dataset class
- [ ] Class weight calculation for imbalanced data

**model.py**
- [ ] Unified function to load any transformer model
- [ ] Support for custom classification heads
- [ ] Class weight integration
- [ ] Focal loss implementation

**metrics.py**
- [ ] Compute all evaluation metrics (accuracy, precision, recall, F1, FNR)
- [ ] Confusion matrix generation
- [ ] Statistical significance tests (McNemar's test)

#### Day 6-7: Baseline Pipeline Validation
**CRITICAL: Validate infrastructure with simple baseline first**

- [ ] Implement TF-IDF + Logistic Regression in `train_baseline.py`
- [ ] Integrate wandb (initialize project, log metrics)
- [ ] Train baseline on 80/10/10 split
- [ ] Verify end-to-end pipeline works
- [ ] Check wandb dashboard shows metrics correctly

**train.py (First Version)**
- [ ] Load config file
- [ ] Load dataset with tokenization
- [ ] Initialize model
- [ ] Training loop with Hugging Face Trainer
- [ ] Evaluation on val/test sets
- [ ] Save checkpoints to Google Drive
- [ ] Log everything to wandb

- [ ] Train DistilBERT with ONE seed using new infrastructure
- [ ] Compare with baseline in wandb
- [ ] Verify model checkpoints save correctly

**Success Criteria:**
- TF-IDF baseline completes successfully
- DistilBERT beats baseline
- wandb shows both runs
- Can load and test saved model

---

### PHASE 2: Systematic Model Training (Week 2-6)

#### Week 2: Class Imbalance Handling
- [ ] Implement class weights (primary approach)
- [ ] Implement focal loss (secondary approach)
- [ ] Test both on DistilBERT with 3 seeds
- [ ] Compare performance, choose best approach
- [ ] Document findings

#### Week 2-3: Core Models (3 seeds each)
Each model trained with:
- Learning rate: 2e-5
- Batch size: 16
- Epochs: 3
- Max sequence length: 128
- Best loss function from previous step
- Save checkpoints to Google Drive

**Training Order:**
1. [ ] DistilBERT (baseline) - 3 seeds
2. [ ] BERT-base - 3 seeds
3. [ ] RoBERTa-base - 3 seeds
4. [ ] ELECTRA-base - 3 seeds (priority)

**After first 4 models:**
- [ ] Analyze results in wandb
- [ ] Decide: Continue with XLNet/DeBERTa OR start hyperparameter tuning?

5. [ ] XLNet-base - 3 seeds (if promising)
6. [ ] DeBERTa-v3-base - 3 seeds (if promising)

**Week 4-6: Alternative Models (Optional)**
- [ ] Simple LSTM baseline
- [ ] TF-IDF + SVM

**Deliverable:** Results table with mean ± std for all metrics across all models

---

### PHASE 3: Hyperparameter Tuning (Week 7-8)

**Select Top 2-3 Models from Phase 2**

#### Hyperparameter Grid
```yaml
learning_rate: [1e-5, 2e-5, 5e-5]
batch_size: [16, 32]
num_epochs: [2, 3, 4]
max_seq_length: [128, 256]
warmup_ratio: [0.06, 0.1]
```

**Strategy:**
- Random search (more efficient than grid search)
- ~15-20 configurations per model
- Train each config with single seed first
- Best configs: retrain with 3 seeds for final results

**Experiments per model:** ~20 configs × 1 run = 20 experiments
**Total:** ~40-60 experiments

- [ ] Create hyperparameter config files
- [ ] Implement random search or use Optuna for optimization
- [ ] Track all experiments in wandb
- [ ] Identify best configuration for each model

**Deliverable:** Best hyperparameters for top 2-3 models

---

### PHASE 4: Ensemble Methods (Week 9)

**Use Top 3 Models from Previous Phases**

#### Ensemble Approaches (in order of complexity)
1. [ ] **Soft Voting** - Average predicted probabilities
   - Simple implementation
   - Baseline ensemble method

2. [ ] **Weighted Averaging** - Weight by validation F1
   - Calculate weights from val performance
   - Optimize weight combinations

3. [ ] **Stacking** - Meta-model approach
   - Use predictions from base models as features
   - Train logistic regression meta-model
   - Cross-validation to generate meta-features

**Evaluation:**
- [ ] Compare ensemble vs individual models
- [ ] Statistical significance testing
- [ ] Check if ensemble reduces FNR

**Deliverable:** Best ensemble configuration

---

### PHASE 5: Error Analysis & Interpretability (Week 10)

#### Quantitative Analysis
- [ ] Generate confusion matrices for all best models
- [ ] Create error buckets:
  - High-confidence False Negatives (FN)
  - Low-confidence False Negatives
  - High-confidence False Positives (FP)
  - Low-confidence False Positives

#### Qualitative Analysis (Focus on False Negatives)
- [ ] Randomly sample 50-100 FN examples from best model
- [ ] Manual linguistic analysis:
  - Identify recurring themes/patterns
  - Look for: sarcasm, conditional statements, passive voice, reported speech
  - Document specific linguistic phenomena
- [ ] Compare FN patterns across different models

#### Model Interpretability
- [ ] Select 10-15 representative FN examples
- [ ] Apply SHAP to show feature importance
- [ ] Apply LIME for local explanations
- [ ] Generate visualizations highlighting influential words
- [ ] Document insights on why models fail

**Deliverable:**
- Error analysis report
- SHAP/LIME visualization notebook
- Linguistic pattern taxonomy

---

### PHASE 6: Statistical Validation & Documentation (Week 11-12)

#### Statistical Testing
- [ ] McNemar's test: Best model vs TF-IDF baseline
- [ ] McNemar's test: Best model vs DistilBERT baseline
- [ ] Paired t-tests between top models (if multiple seeds)
- [ ] Report p-values and effect sizes
- [ ] Determine statistical significance (p < 0.05)

#### Comprehensive Evaluation
- [ ] Compile all results into master table
- [ ] Generate comparison visualizations:
  - Bar charts (accuracy, F1, precision, recall, FNR)
  - Precision-recall curves
  - ROC curves
  - Confusion matrices (heatmaps)
  - Training time comparisons
  - Model size vs performance

#### Documentation & Reporting
- [ ] **Methodology Section:**
  - Dataset description and preprocessing
  - Model architectures and configurations
  - Training procedures and hyperparameters
  - Evaluation metrics and protocols

- [ ] **Results Section:**
  - Performance tables with mean ± std
  - Statistical significance results
  - Best configurations for each model
  - Ensemble method comparisons

- [ ] **Discussion Section:**
  - Error analysis findings
  - Interpretability insights
  - Practical recommendations
  - Limitations and future work

- [ ] **Ethical Considerations Section:**
  - Data privacy and anonymization
  - Model limitations and appropriate use cases
  - Risk of misuse (not a diagnostic tool)
  - Potential biases in model predictions
  - Importance of human oversight
  - Demographic bias analysis (if demographic data available)

---

## Technical Implementation Details

### Data Preprocessing Strategy
**Tokenization Caching:**
- Tokenize entire dataset once upfront
- Save tokenized datasets to disk (use HuggingFace `datasets.save_to_disk()`)
- Create separate cached versions for max_seq_length [128, 256]
- Avoids bottleneck during training

**Class Imbalance Handling:**
1. Calculate class weights: `weight = n_samples / (n_classes * class_count)`
2. Pass to Trainer via `compute_loss` override OR
3. Implement focal loss as alternative

### Experiment Tracking with wandb

```python
import wandb

# Initialize
wandb.init(
    project="suicide-ideation-detection",
    config={
        "model_name": "bert-base",
        "learning_rate": 2e-5,
        "batch_size": 16,
        "epochs": 3,
        "seed": 42,
        # ... all hyperparameters
    }
)

# Log metrics during training
wandb.log({
    "train_loss": loss,
    "val_f1": f1_score,
    "val_accuracy": accuracy,
    # ...
})

# Log final results
wandb.log({
    "test_f1": test_f1,
    "test_fnr": test_fnr,
    # ...
})
```

### Reproducibility Checklist
- [ ] All random seeds documented and set (Python, NumPy, PyTorch, Transformers)
- [ ] Data split indices saved and version-controlled
- [ ] All config files version-controlled
- [ ] Model checkpoints saved with metadata (config, seed, timestamp)
- [ ] Requirements.txt with exact package versions
- [ ] Training scripts logged to wandb
- [ ] Google Drive backup of all checkpoints

---

## Google Colab Pro Optimization

### Resource Management
```python
# Enable mixed precision (2x speedup)
training_args = TrainingArguments(
    fp16=True if torch.cuda.is_available() else False,
    # ...
)

# Clear CUDA cache between experiments
import gc
gc.collect()
torch.cuda.empty_cache()

# Gradient checkpointing for larger models
model.gradient_checkpointing_enable()
```

### Save to Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')

# Save checkpoints to Drive
output_dir = '/content/drive/MyDrive/suicide-detection/results'
```

### Session Management
- Use 24hr high-RAM GPU runtime
- Set up checkpoint saving every N steps
- If session disconnects, resume from last checkpoint
- Download critical results immediately after training

---

## Expected Timeline (Detailed)

| Phase | Duration | Key Deliverables |
|-------|----------|------------------|
| Phase 0: Data Validation | 2 days | Split indices, EDA notebook |
| Phase 1: Infrastructure | 5 days | Modular codebase, wandb integration, baseline |
| Phase 2: Model Training | 4-5 weeks | 6 models × 3 seeds, performance table |
| Phase 3: Hyperparameter Tuning | 2 weeks | Optimal configs for top models |
| Phase 4: Ensemble Methods | 1 week | Best ensemble approach |
| Phase 5: Error Analysis | 1 week | Error patterns, interpretability report |
| Phase 6: Validation & Writing | 1-2 weeks | Statistical tests, paper draft |
| **TOTAL** | **7-12 weeks** | Complete research paper |

---

## Key Success Metrics

### Technical Success
- [ ] All models beat TF-IDF baseline with statistical significance
- [ ] Best model achieves F1 > 0.90
- [ ] False Negative Rate < 5%
- [ ] Results reproducible across seeds (low std)

### Research Success
- [ ] Clear ranking of model architectures
- [ ] Actionable insights from error analysis
- [ ] Novel linguistic patterns identified
- [ ] Ethical considerations thoroughly addressed

### Publication Success
- [ ] Comprehensive methodology documented
- [ ] Statistical significance established
- [ ] Reproducibility guaranteed (code + configs shared)
- [ ] Domain-specific contribution (beyond pure benchmarking)

---

## Risk Mitigation

### Technical Risks
| Risk | Mitigation |
|------|------------|
| Colab session timeout | Checkpoint every 500 steps, save to Drive |
| GPU memory issues | Gradient accumulation, smaller batch sizes, fp16 |
| Data leakage | Verify split stratification, no test set touching |
| Poor model performance | Start with strong baselines, validate pipeline early |

### Research Risks
| Risk | Mitigation |
|------|------------|
| No significant difference between models | Focus on error analysis for insights |
| High false negative rate | Prioritize recall, use class weights/focal loss |
| Non-reproducible results | Multiple seeds, document all randomness sources |

---

## Next Immediate Steps (This Week)

### Day 1 (Today)
1. Run EDA on `Suicide_Detection_Full.csv`
2. Check class distribution
3. Analyze text length distribution

### Day 2
4. Implement and verify 80/10/10 split
5. Save split indices
6. Confirm stratification worked

### Day 3-4
7. Set up project directory structure
8. Create config system
9. Refactor existing code into modular format

### Day 5
10. Implement TF-IDF baseline
11. Set up wandb account and project
12. Run baseline to validate pipeline

### Weekend
13. Review results
14. Begin transformer infrastructure (dataset.py, model.py)

---

## Open Questions to Resolve

1. **Data Augmentation:** Should we implement back-translation or synonym replacement given likely class imbalance?
2. **Sequence Length:** Most texts < 256 tokens, but should we test 512 for completeness?
3. **Additional Models:** Should we include RoBERTa-large if smaller models show promise?
4. **Ensemble Complexity:** Is stacking worth the complexity vs weighted averaging?
5. **Demographic Analysis:** Do we have demographic metadata to check for bias?

---

## References & Resources

### Key Papers to Review
- BERT: Devlin et al. (2019)
- RoBERTa: Liu et al. (2019)
- ELECTRA: Clark et al. (2020)
- DeBERTa: He et al. (2021)
- Focal Loss: Lin et al. (2017)

### Tools & Libraries
- Transformers (HuggingFace)
- PyTorch
- Scikit-learn
- Weights & Biases (wandb)
- SHAP / LIME

---

## Appendix: Configuration File Examples

### base_config.yaml
```yaml
# Data
data:
  dataset_path: "data/Suicide_Detection_Full.csv"
  split_indices_path: "data/splits/split_indices.json"
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1
  random_seed: 42

# Training
training:
  num_train_epochs: 3
  per_device_train_batch_size: 16
  per_device_eval_batch_size: 32
  learning_rate: 2.0e-5
  weight_decay: 0.01
  warmup_ratio: 0.1
  fp16: true
  gradient_accumulation_steps: 1

# Evaluation
evaluation:
  eval_strategy: "steps"
  eval_steps: 500
  save_strategy: "steps"
  save_steps: 500
  logging_steps: 100
  load_best_model_at_end: true
  metric_for_best_model: "f1"

# Model
model:
  max_seq_length: 128
  num_labels: 2

# Experiment Tracking
wandb:
  project: "suicide-ideation-detection"
  entity: "your-username"

# Reproducibility
seeds: [42, 123, 456]
```

### distilbert.yaml
```yaml
model:
  model_name: "distilbert-base-uncased"
  tokenizer_name: "distilbert-base-uncased"
  model_type: "distilbert"
```

---

**Document Version:** 1.0
**Last Updated:** 2025-10-04
**Created by:** Claude + Gemini Collaborative Planning
**Status:** Ready for Implementation
