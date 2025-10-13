from celery import Celery, Task, states
from celery.exceptions import Ignore
import time
import logging
import importlib
import sys
from pathlib import Path
import redis
import os

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from celery_src.utils import run_attack

benchmarking = importlib.import_module("benchmarking")

# Configure Celery to use Redis as the message broker
celery = Celery(
    "worker",  # This is the name of your Celery application
    broker="redis://localhost:6379",  # This is the Redis connection string
    backend="redis://localhost:6379/0",  # Optional, for storing task results
    task_track_started = True,
    task_send_sent_event = True,
    worker_send_task_events = True,
)
celery.conf.broker_connection_retry_on_startup = True

redis_client = redis.StrictRedis(host=os.environ.get('REDIS_HOST','localhost'), 
                                     port=os.environ.get('REDIS_PORT',6379), 
                                     db=0, 
                                     decode_responses=True)

class CancelableTask(Task):
    def is_cancelled(self):
        task_id = self.request.id
        return redis_client.get(f"cancel_flag:{task_id}") == "1"


@celery.task(bind=True, name="benchmarking-task", base=CancelableTask)
def benchmarking_task(self: Task, benchmark_input: dict):
    return benchmarking.benchmark(benchmark_input)

run_attack_task = celery.task(run_attack, name="single-image-attack-task")
