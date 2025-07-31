from celery import Celery
import time
import logging

# Configure Celery to use Redis as the message broker
celery = Celery(
    "worker",  # This is the name of your Celery application
    broker="redis://localhost:6379",  # This is the Redis connection string
    #backend="redis://localhost:6379/0",  # Optional, for storing task results
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
