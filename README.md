# Mental Health Ideation Detection

A machine-learning project for classifying mental-health risk signals in text. The repository contains two connected parts:

- a reproducible research pipeline for training and evaluating transformer models
- a deployed FastAPI application for running inference, collecting reviewer feedback, and monitoring model behavior

Live application: https://mentalhealthideation.com

## Deployment

Since September 2025 the application has been running at James Logan High School (enrollment about 4,000 students). It has surfaced nearly 100 potentially concerning submissions for educator review, and 5 teachers rated its usefulness 3.7/5. Every flag is reviewed by a person; the model is decision support, not a diagnosis.

## Results

The current DistilBERT run uses a 232,074-sample balanced text dataset with an 80/10/10 train/validation/test split.

| Metric | Test result |
| --- | ---: |
| Accuracy | 97.96% |
| Precision | 98.24% |
| Recall | 97.66% |
| F1 | 97.95% |
| False-negative rate | 2.34% |

The saved run configuration uses a learning rate of `2e-5`, batch size 16, 3 epochs, and a maximum sequence length of 256.

See `research/results/distilbert-seed42/results.json` for the complete output.

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

The current production model is DistilBERT. Additional model configs are included for BERT, RoBERTa, and ELECTRA.

### Run training

```bash
cd research
pip install -r requirements-research.txt
python src/train.py \
  --base-config config/base_config.yaml \
  --model-config config/model_configs/distilbert.yaml
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
