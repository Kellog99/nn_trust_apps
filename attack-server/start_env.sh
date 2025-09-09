#!/usr/bin/env bash
# This script take care of all preliminary operation and start all services
# using no hup and storing PIDs, trapping the SIGNINT signal is able to stop all started process in background when using ctrl + c
# Stdout and stderr of backgorund processes are logged to separate files in the ./logs folder
mkdir -p ./logs
source .venv/bin/activate
mkdir -p submodules/data-quality_gui/public/titann/datasets
mkdir -p submodules/data-quality_gui/public/titann/models

set -e
PIDS=()

trap 'echo "Stopping all services..."; kill ${PIDS[@]}; docker rm -f redis_app; exit 0' SIGINT SIGTERM

nohup bash -c 'docker run -p 6379:6379 --name redis_app redis' > logs/redis_app.log 2>&1 &
PIDS+=("$!")
nohup bash -c 'source .venv/bin/activate && celery -A celery_worker.celery --workdir ./celery_src worker --pool=eventlet -n=worker1' > logs/celery_worker-1.log 2>&1 &
PIDS+=("$!")
nohup bash -c 'source .venv/bin/activate && celery -A celery_worker.celery --workdir ./celery_src worker --pool=eventlet -n=worker2' > logs/celery_worker-2.log 2>&1 &
PIDS+=("$!")
nohup bash -c 'source .venv/bin/activate && celery -A celery_worker.celery --workdir ./celery_src flower --port=5555' > logs/flower.log 2>&1 &
PIDS+=("$!")
nohup bash -c 'cd submodules/data-quality_gui && npm run dev' > logs/npm_dev.log 2>&1 &
PIDS+=("$!")
source .venv/bin/activate && python app.py

wait