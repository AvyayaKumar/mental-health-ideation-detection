# Deployment

This directory contains the web application and supporting services for running the trained classifier.

## Components

- **FastAPI** application for inference and feedback endpoints
- **PostgreSQL** for predictions, reviewer feedback, and analytics data
- **Celery + Redis** for queued background work
- **SQLAlchemy** for database access
- **Jinja2** templates for the web and administrative interfaces
- **Docker / Docker Compose** for local development
- **Railway** configuration for hosted deployment

## Local setup

Create an environment file:

```bash
cp .env.example .env
```

Then start the services:

```bash
docker-compose up
```

The FastAPI service is available at:

```text
http://localhost:8000
```

Interactive API documentation is provided by FastAPI at `/docs`.

## Backend structure

```text
deployment/
├── backend/
│   ├── app.py
│   ├── worker.py
│   ├── scheduler.py
│   ├── models/
│   │   ├── database.py
│   │   ├── model_loader.py
│   │   ├── pipeline.py
│   │   └── interpretability.py
│   ├── services/
│   └── templates/
├── scripts/
├── docker-compose.yml
└── .env.example
```

## API

The application includes endpoints for:

- synchronous prediction
- queued analysis
- retrieving queued results
- submitting reviewer feedback
- reviewing stored feedback
- basic health checks

See `backend/app.py` for the current route definitions.

## Model loading

The deployed application serves the RoBERTa seed-42 run and falls back to the project's rule-based scorer if the model can't be loaded.

On Railway, `deployment/.railway/Dockerfile` downloads the model zip from Google Drive at build time (`GDRIVE_FILE_ID`). The zip must contain `backend/models/<run>/final_model/`, where `<run>` is the `MODEL_RUN` build arg (default `roberta-seed42`). The build also copies `research/results/<run>/results.json` next to the model and saves the base tokenizer into the model folder. `GET /api/v1/model-info` and every prediction response report the model name and its held-out test metrics from that file; the web UI shows them.

## Feedback and retraining

Reviewer corrections are stored in the database and can be exported or incorporated into retraining workflows using the scripts in this directory.

The retraining utilities are intentionally separated from the request path so model updates do not block normal application traffic.

## Privacy

Before model analysis the application redacts email addresses and phone numbers (`backend/services/pii.py`); it doesn't detect names or other personal information. Raw user text should not be treated as durable analytics data unless explicitly required and appropriately protected.

This project is a research and decision-support tool, not a diagnostic system. Predictions require human review.
