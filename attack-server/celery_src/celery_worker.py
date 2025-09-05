from celery import Celery
import time
import logging
import importlib
import sys
from pathlib import Path

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


@celery.task()
def sum_celery_task(x,y,d):
    time.sleep(15)
    with open("log_celery.txt", "a") as f:
        f.write(f"[{d}] Sum of {x}, {y} is {x+y}\n")
    logging.info("*"*30)
    logging.info(f"JOb completed, results written to file")
    return "Test return task"

benchmarking_task = celery.task(benchmarking.benchmark_, name="benchmarking-task", bind=True)
run_attack_task = celery.task(run_attack, name="single-image-attack-task")
