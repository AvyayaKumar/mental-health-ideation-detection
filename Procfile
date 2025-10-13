web: cd deployment/backend && uvicorn app:app --host 0.0.0.0 --port $PORT
worker: cd deployment/backend && celery -A worker.celery_app worker --loglevel=info --concurrency=2
