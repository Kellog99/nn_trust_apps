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
BENCHMARK_OUTPUT_DIR=${BENCHMARK_OUTPUT_DIR:-benchmark_results}

echo "Creating directories:"
echo "  Model Repository: $MODEL_REPO"
echo "  Datasets Repository: $DATASETS_REPO"
echo "  Benchmark Output: $BENCHMARK_OUTPUT_DIR"
mkdir -p "$MODEL_REPO"
mkdir -p "$DATASETS_REPO"
mkdir -p "$BENCHMARK_OUTPUT_DIR"

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
    
    # Stop Ray cluster
    if command -v ray &> /dev/null; then
        echo "Stopping Ray cluster..."
        ray stop 2>/dev/null || true
    fi
    
    # Kill background processes
    if [ ${#PIDS[@]} -gt 0 ]; then
        kill ${PIDS[@]} 2>/dev/null || true
    fi
    
    # Stop Redis if running
    docker rm -f redis_app 2>/dev/null || true
    
    echo "All services stopped"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Set Ray configuration with defaults
RAY_HEAD_HOST=${RAY_HEAD_HOST:-127.0.0.1}
RAY_HEAD_PORT=${RAY_HEAD_PORT:-6379}
RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8265}
RAY_NUM_ACTORS=${RAY_NUM_ACTORS:-4}

echo "Ray Cluster Configuration:"
echo "  Head Host: $RAY_HEAD_HOST"
echo "  Head Port: $RAY_HEAD_PORT"
echo "  Dashboard Port: $RAY_DASHBOARD_PORT"
echo "  Number of Actors: $RAY_NUM_ACTORS"

# Start Ray cluster on head node
echo "Starting Ray cluster head node..."
ray start --head \
    --node-ip-address="$RAY_HEAD_HOST" \
    --port="$RAY_HEAD_PORT" \
    --dashboard-port="$RAY_DASHBOARD_PORT" \
    > ./logs/ray_cluster.log 2>&1 &

RAY_PID=$!
PIDS+=($RAY_PID)
echo "Ray cluster started with PID: $RAY_PID"

# Wait a moment for services to start
sleep 3

echo "All background services started. PIDs: ${PIDS[*]}"
echo "Logs available in ./logs/ directory"
echo "Services running:"
echo "  - Ray Cluster: http://$RAY_HEAD_HOST:$RAY_DASHBOARD_PORT (Dashboard)"
echo "  - Ray Head: $RAY_HEAD_HOST:$RAY_HEAD_PORT"

# Start main application (foreground)
echo "Starting main application..."

# Set application variables with defaults
APP_HOST=${HOST:-0.0.0.0}
APP_PORT=${PORT:-8000}
APP_WORKERS=${WORKERS:-1}
MAX_MODEL_SIZE=${MAX_MODEL_SIZE_UPLOAD:-5000}
MAX_MODEL_JSON_SIZE=${MAX_MODEL_JSON_SIZE_UPLOAD:-5000}
RAY_ADDRESS="${RAY_HEAD_HOST}:${RAY_HEAD_PORT}"
RAY_PY_MODS=${RAY_PY_MODULES:-}

echo "Application Configuration:"
echo "  Host: $APP_HOST"
echo "  Port: $APP_PORT" 
echo "  Workers: $APP_WORKERS"
echo "  Model Storage: $MODEL_REPO"
echo "  Dataset Storage: $DATASETS_REPO"
echo "  Benchmark Output: $BENCHMARK_OUTPUT_DIR"
echo "  Max Model Upload Size: $MAX_MODEL_SIZE MB"
echo "  Max Model JSON Upload Size: $MAX_MODEL_JSON_SIZE MB"
echo "  Ray Address: $RAY_ADDRESS"
if [ -n "$RAY_PY_MODS" ]; then
    echo "  Ray Python Modules: $RAY_PY_MODS"
fi

source .venv/bin/activate && python app.py \
    --host "$APP_HOST" \
    --port "$APP_PORT" \
    --workers "$APP_WORKERS" \
    --ds_storage "$DATASETS_REPO" \
    --model_storage "$MODEL_REPO" \
    --benchmark_output_dir "$BENCHMARK_OUTPUT_DIR" \
    --max_model_size_upload "$MAX_MODEL_SIZE" \
    --max_model_json_size_upload "$MAX_MODEL_JSON_SIZE" \
    --ray_address "$RAY_ADDRESS" \
    ${RAY_PY_MODS:+--ray_py_modules "$RAY_PY_MODS"}

# Wait for all background processes
wait