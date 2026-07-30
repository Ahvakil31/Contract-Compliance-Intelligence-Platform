# config.py
import os
from celery import Celery

# Central environmental definitions
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

# Celery application instantiation
celery_app = Celery(
    "legal_nlp_tasks",
    broker=REDIS_URL,
    backend=REDIS_URL
)

# Optimize celery for resource-heavy machine learning workflows
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    
    # WINDOWS & THREAD POOL OPTIMIZATIONS:
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=50,
    
    # OCR PIPELINE TIMEOUT SAFEGUARDS:
    task_time_limit=600,
    task_soft_time_limit=540
)

# Model configuration constants
LEGAL_BERT_MODEL = "nlpaueb/legal-bert-base-uncased"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
SPACY_MODEL = "en_core_web_sm"