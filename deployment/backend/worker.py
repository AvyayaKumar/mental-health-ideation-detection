import os
from pathlib import Path
from celery import Celery, signals
from dotenv import load_dotenv
from models.pipeline import AnalysisPipeline

# Load .env file from parent directory
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    __name__,
    broker=REDIS_URL,
    backend=REDIS_URL,
)

# Global pipeline - will be initialized in worker process
_pipeline = None

@signals.worker_process_init.connect
def init_worker(**kwargs):
    """Initialize worker process - load env and create pipeline."""
    global _pipeline
    # Reload .env in worker process
    load_dotenv(dotenv_path=env_path, override=True)

    # Force reload the model_loader with new environment
    from models.model_loader import model_loader
    model_loader._model = None
    model_loader._tokenizer = None
    model_loader._load_model()

    # Create pipeline (this will use the reloaded model)
    _pipeline = AnalysisPipeline()


@celery_app.task(name="analyze_text", time_limit=60, soft_time_limit=45)
def analyze_text(text: str):
    """
    Run analysis pipeline on text. Returns JSON-serializable dict.

    Timeout: 60s hard limit, 45s soft limit (for Railway compatibility)
    """
    return _pipeline.analyze(text)
