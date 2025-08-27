#!/usr/bin/env bash

set -e
PIDS=()

trap 'echo "Stopping all services..."; kill ${PIDS[@]}; docker kill redis; exit 0' SIGINT SIGTERM

nohup bash -c 'docker run -p 6379:6379 --name redis redis' > logs/redis_app.log 2>&1 &
PIDS+=("$!")
nohup bash -c 'source .venv/bin/activate && celery -A celery_worker.celery --workdir ./celery worker' > logs/celery_worker.log 2>&1 &
PIDS+=("$!")
nohup bash -c 'source .venv/bin/activate && celery -A celery_worker.celery --workdir ./celery flower' > logs/flower.log 2>&1 &
PIDS+=("$!")
mkdir -p submodules/data-quality_gui/public/titann/datasets &
mkdir -p submodules/data-quality_gui/public/titann/models &
nohup bash -c 'cd submodules/data-quality_gui && npm run dev' > logs/npm_dev.log 2>&1 &
PIDS+=("$!")
source .venv/bin/activate && python app.py

wait