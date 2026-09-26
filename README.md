# Mental Health Ideation Detection

A machine-learning project for classifying mental-health risk signals in text. The repository contains two connected parts:

- a reproducible research pipeline for training and evaluating transformer models
- a deployed FastAPI application for running inference, collecting reviewer feedback, and monitoring model behavior

Live application: https://mentalhealthideation.com

## What the 99.45% does and doesn't mean

The deployed RoBERTa model scores 99.45% F1 on a held-out test set. That number is real and reproducible, but it measures how well the model separates two groups of Reddit posts, not how well it screens student writing. The short version: the test set is much easier than the job, and on school-style sentences the model misses indirect warning signs. Details below; every number in this section comes from `research/scripts/audit_evaluation.py` and is saved in `research/results/audit/audit.json`.

### Dataset

- 232,074 Reddit posts, exactly balanced (116,037 per class). This is the public Kaggle "Suicide and Depression Detection" dataset (`Suicide_Detection.csv`). It isn't committed here.
- Labels come from the subreddit a post was made in, not from a human reading it. Per the dataset's description, `suicide` posts are from r/SuicideWatch and r/depression, and `non-suicide` posts are from r/teenagers.
- That makes the task easier than it sounds. The strongest TF-IDF features for the negative class are words like *teenagers, crush, minecraft, discord, bruh, karma, filler, memes*: the model can win by recognizing r/teenagers chatter, without understanding risk.
- There's also a formatting shortcut: 20.4% of `suicide` posts have line breaks stripped so sentences run together ("...to meI can't"), versus 0.03% of `non-suicide` posts.
- Because labels are per subreddit, some are wrong in both directions: r/teenagers posts that state a plan to die are labeled `non-suicide`, and r/depression posts about unrelated things (a concert ticket, a doctor's visit) are labeled `suicide`.

### Split

- One fixed, stratified, random 80/10/10 split at the post level, seed 42: 185,659 train / 23,207 validation / 23,208 test (11,604 per class in test). Indices are in `research/data/splits/split_indices.json`, and every model uses the same ones.
- Every model used the same hyperparameters (learning rate `2e-5`, batch size 16, 3 epochs, max length 256), with no per-model tuning. RoBERTa is also the best model on the validation set (99.51% F1 vs 97.91–98.21% for the others), so choosing it doesn't depend on the test set.
- The dataset has no author IDs, so the split can't be grouped by author. Posts by the same person can land on both sides, which could inflate scores somewhat.

### Duplicate handling

- Training drops empty rows and nothing else; there's no dedup step. So I checked for leakage directly.
- 0 exact duplicate texts. 93 rows duplicate another row after lowercasing and collapsing whitespace, and none of those pairs disagree on label.
- 11 of 23,208 test rows (0.05%) and 15 validation rows also appear in train after that normalization. None of the model's test errors are in those rows, so removing them wouldn't change the result.

### Baseline and benchmark

All numbers are on the held-out test set (23,208 posts).

| Model | Test accuracy | Test F1 | FNR | Mean test accuracy (3 seeds) |
| --- | ---: | ---: | ---: | ---: |
| TF-IDF + logistic regression | 93.57% | 93.51% | 7.26% | n/a (deterministic) |
| DistilBERT (seed 42) | 97.96% | 97.95% | 2.34% | 97.94% |
| BERT (seed 42) | 97.94% | 97.94% | 1.76% | 97.98% |
| ELECTRA (seed 42) | 98.26% | 98.26% | 1.71% | 98.23% |
| **RoBERTa (seed 42, deployed)** | **99.45%** | **99.45%** | **0.75%** | 99.43% |

The baseline uses 10,000 unigram/bigram features and `class_weight="balanced"` (`research/scripts/train_baseline.py`). A bag of words already gets 93.5%, which says most of the separation comes from vocabulary. Each transformer run's full output is in `research/results/<model>-seed<seed>/results.json`, with summaries in `results_summary.csv` and `aggregated_results.csv`. The app reads the deployed run's `results.json` and shows that model name and test accuracy in the UI, so the label can't drift from the file.

### Failure cases

On the test set, RoBERTa makes 127 errors: 87 false negatives and 40 false positives (precision 99.65%, recall 99.25%). Reading through them:

- **Many false positives are label noise.** A lot of them are r/teenagers posts describing self-harm or a plan to die. The model is arguably right and the label is wrong. The same goes for false negatives like a thank-you note to r/SuicideWatch or a concert-ticket post in r/depression.
- **The real misses** are non-English posts, ideation wrapped in jokes or "lol", misspelled or drunk writing, passive wishes ("I wish a car would hit me"), and posts about researching methods that never use the words *suicide* or *kill myself*.
- **The errors are confident.** 79 of the 87 false negatives got a suicide probability under 0.05, and only 0.1% of all test predictions fall between 0.1 and 0.9. The probability isn't a usable "unsure" signal.

**Out-of-domain probes.** The test set has no student writing, so I ran 12 short sentences written to look like school work (they're in the audit script). That's far too few to estimate a rate; it only shows what kind of input fails:

| Expected | Model output | Example |
| --- | --- | --- |
| not concerning | correct on 5/5 | "This chemistry test is going to kill me", an essay about Romeo and Juliet, an argumentative essay about teen suicide rates |
| concerning, explicit | caught 1 of 2 | "I don't want to be alive anymore" → 0.91; "I want to kill myself." → **0.10 (missed)** |
| concerning, indirect | caught 0 of 5 | "everyone would be better off without me", "go to sleep and not wake up", giving belongings away, a narrative about writing a goodbye letter, "i kinda want to die haha" |

The model catches explicit, Reddit-length statements of intent and misses indirect warning signs, which are exactly what a teacher most needs help noticing. The live app uses this model's prediction directly (the keyword rules in `services/risk.py` only run if the model fails to load), so these limits apply to the deployment.

## Deployment at James Logan High School

Since September 2025 the application has been used by teachers at James Logan High School. What exists so far:

- It has flagged nearly 100 submissions for educator review.
- It averaged 3.7/5 across a multi-question usefulness survey of 5 teachers.

What doesn't exist yet: labeled results on school submissions. There's no count of how many flags teachers judged correct, and no way to know how many concerning essays were never flagged, because unflagged essays aren't reviewed. Given the probe results above, the model should be treated as a narrow backstop for explicit language, not as a screen that clears an essay. Every flag is reviewed by a person; the model is decision support, not a diagnosis.

The next step is measuring against teacher judgment on real (redacted) student writing, including a sample of unflagged essays, before claiming any accuracy on the school task.

## Stack

**Research:** Python, PyTorch, Hugging Face Transformers, scikit-learn, Weights & Biases

**Backend:** FastAPI, SQLAlchemy, PostgreSQL, Celery, Redis

**Deployment:** Docker, Docker Compose, Railway, Uvicorn

**Interpretability:** Integrated Gradients, with additional analysis utilities for model explanations

## Repository structure

```text
.
├── research/
│   ├── config/       # shared and model-specific training configs
│   ├── src/          # datasets, models, metrics, and training code
│   ├── scripts/      # baselines, retraining, and utility scripts
│   └── results/      # saved experiment outputs
├── deployment/
│   ├── backend/      # FastAPI service and model pipeline
│   ├── scripts/      # feedback export and retraining utilities
│   └── docker-compose.yml
└── docs/
    ├── EDA_SUMMARY.md
    └── RESEARCH_PLAN.md
```

## Research pipeline

The research code supports:

- configurable transformer training
- reproducible random seeds and saved experiment metadata
- validation and held-out test evaluation
- accuracy, precision, recall, F1, false-negative rate, and false-positive rate
- classical TF-IDF + logistic-regression baselines
- model interpretability experiments
- experiment logging with Weights & Biases

The deployed model is RoBERTa (seed 42). Configs for DistilBERT, BERT, RoBERTa, and ELECTRA are in `research/config/model_configs/`.

### Run training

```bash
cd research
pip install -r requirements-research.txt
python src/train.py \
  --base_config config/base_config.yaml \
  --model_config config/model_configs/distilbert.yaml
```

See `research/README.md` for the research workflow.

## Application

The application stores reviewer feedback in PostgreSQL and includes an administrative interface for reviewing predictions and feedback.

- **Live demo:** the web UI calls the synchronous `POST /api/v1/predict` endpoint with `explain=False`, so predictions come back in a single request.
- **In the code but off in the demo:** a Celery/Redis queue (`POST /api/v1/analyze` plus `GET /api/v1/result/{task_id}`) and Integrated Gradients word highlighting. IG is CPU-heavy, so it's disabled on the Railway deployment.

Docker Compose provides a local multi-service setup, including Celery and Redis.

### Run locally

```bash
git clone https://github.com/AvyayaKumar/mental-health-ideation-detection.git
cd mental-health-ideation-detection/deployment
cp .env.example .env
docker-compose up
```

The application is then available at `http://localhost:8000`.

See `deployment/README.md` for deployment details.

## Dataset and privacy

The training dataset itself is not committed to this repository. The repository also does not include student essays or other production user text.

Before analysis, the application redacts email addresses and phone numbers (`deployment/backend/services/pii.py`). It doesn't detect other personal information such as names or addresses. This project is intended as a research and decision-support system, not a diagnostic tool, and model output should always be reviewed by a person in context.

## License

MIT
