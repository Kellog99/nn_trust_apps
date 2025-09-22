#!/usr/bin/env bash
# This script takes care of all preliminary operations and starts all services
# using nohup and storing PIDs, trapping the SIGINT signal to stop all started processes in background when using ctrl + c
# Stdout and stderr of background processes are logged to separate files in the ./logs folder

set -e  # Exit on any error

# Load environment variables
ENV=${ENVIRONMENT:-dev}
echo "Starting services for environment: $ENV"

if [ -f ".env.${ENV}" ]; then
    echo "Loading environment variables from .env.${ENV}"
    # Export all non-comment, non-empty lines from the env file
    export $(grep -v '^#' ".env.${ENV}" | grep -v '^$' | xargs)
else
    echo "Warning: Environment file .env.${ENV} not found, using defaults"
fi

# Create necessary directories
mkdir -p ./logs

# Create model and dataset repositories from environment variables
MODEL_REPO=${MODEL_REPO:-models_repo}
DATASETS_REPO=${DATASETS_REPO:-datasets_repo}

echo "Creating directories:"
echo "  Model Repository: $MODEL_REPO"
echo "  Datasets Repository: $DATASETS_REPO"

mkdir -p "$MODEL_REPO"
mkdir -p "$DATASETS_REPO"

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "Virtual environment activated"
else
    echo "Warning: Virtual environment not found at .venv/bin/activate"
fi

# Initialize PID array and trap
PIDS=()
cleanup() {
    echo "Stopping all services..."
    if [ ${#PIDS[@]} -gt 0 ]; then
        kill ${PIDS[@]} 2>/dev/null || true
    fi
    docker rm -f redis_app 2>/dev/null || true
    echo "All services stopped"
    exit 0
}
trap cleanup SIGINT SIGTERM

# Use environment variables with defaults
REDIS_PORT=${REDIS_PORT:-6379}
FLOWER_PORT=${FLOWER_PORT:-5555}
CELERY_APP=${CELERY_APP:-celery_worker.celery}
CELERY_WORKDIR=${CELERY_WORKDIR:-./celery_src}
GUI_DEV_PORT=${GUI_DEV_PORT:-3000}

echo "Configuration:"
echo "  Redis Port: $REDIS_PORT"
echo "  Flower Port: $FLOWER_PORT"
echo "  Celery App: $CELERY_APP"
echo "  Celery Working Directory: $CELERY_WORKDIR"
echo "  GUI Development Port: $GUI_DEV_PORT"

# Start Redis
echo "Starting Redis..."
nohup bash -c "docker run -p ${REDIS_PORT}:6379 --name redis_app redis" > logs/redis_app.log 2>&1 &
PIDS+=("$!")

# Wait a moment for Redis to start
sleep 2

# Start Celery Workers
echo "Starting Celery workers..."
nohup bash -c "source .venv/bin/activate && celery -A ${CELERY_APP} --workdir ${CELERY_WORKDIR} worker --pool=solo -n=worker1 --loglevel=${CELERY_LOG_LEVEL:-info}" > logs/celery_worker-1.log 2>&1 &
PIDS+=("$!")

nohup bash -c "source .venv/bin/activate && celery -A ${CELERY_APP} --workdir ${CELERY_WORKDIR} worker --pool=solo -n=worker2 --loglevel=${CELERY_LOG_LEVEL:-info}" > logs/celery_worker-2.log 2>&1 &
PIDS+=("$!")

# Start Flower (Celery monitoring)
echo "Starting Flower..."
nohup bash -c "source .venv/bin/activate && celery -A ${CELERY_APP} --workdir ${CELERY_WORKDIR} flower --port=${FLOWER_PORT}" > logs/flower.log 2>&1 &
PIDS+=("$!")

# Start GUI development server
echo "Starting GUI development server..."
nohup bash -c "cd submodules/data-quality_gui && npm run dev -- --port ${GUI_DEV_PORT}" > logs/npm_dev.log 2>&1 &
PIDS+=("$!")

# Wait a moment for services to start
sleep 3

echo "All background services started. PIDs: ${PIDS[*]}"
echo "Logs available in ./logs/ directory"
echo "Services running:"
echo "  - Redis: http://localhost:${REDIS_PORT}"
echo "  - Flower: http://localhost:${FLOWER_PORT}"
echo "  - GUI Dev: http://localhost:${GUI_DEV_PORT}"

# Start main application (foreground)
echo "Starting main application..."

# Set application variables with defaults
APP_HOST=${HOST:-0.0.0.0}
APP_PORT=${PORT:-8000}
APP_WORKERS=${WORKERS:-1}

echo "Application Configuration:"
echo "  Host: $APP_HOST"
echo "  Port: $APP_PORT" 
echo "  Workers: $APP_WORKERS"
echo "  Model Storage: $MODEL_REPO"
echo "  Dataset Storage: $DATASETS_REPO"

source .venv/bin/activate && python app.py \
    --host "$APP_HOST" \
    --port "$APP_PORT" \
    --workers "$APP_WORKERS" \
    --ds_storage "$DATASETS_REPO" \
    --model_storage "$MODEL_REPO"

# Wait for all background processes
wait