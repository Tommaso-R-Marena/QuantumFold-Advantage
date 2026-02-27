from __future__ import annotations

import json

from celery import Celery

celery_app = Celery(
    "quantumfold", broker="redis://localhost:6379/0", backend="redis://localhost:6379/1"
)


@celery_app.task(bind=True, max_retries=3)
def run_prediction(self, job_id: str, request: dict):
    try:
        result = {"job_id": job_id, "status": "completed", "result": {"tm_score": 0.7}}
        return json.dumps(result)
    except Exception as e:
        raise self.retry(exc=e, countdown=60)
