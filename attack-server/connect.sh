#!/usr/bin/env bash
# This script connects a worker node to an existing Ray cluster
# It should be run on worker nodes that have cloned the same repository
# The head node address must be provided as an argument or environment variable

set -e  # Exit on any error

# Load environment variables
ENV=${ENVIRONMENT:-dev}
echo "Connecting worker node for environment: $ENV"

if [ -f ".env.${ENV}" ]; then
    echo "Loading environment variables from .env.${ENV}"
    # Export all non-comment, non-empty lines from the env file
    export $(grep -v '^#' ".env.${ENV}" | grep -v '^$' | xargs)
else
    echo "Warning: Environment file .env.${ENV} not found, using defaults"
fi

# Create necessary directories
mkdir -p ./logs

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "Virtual environment activated"
else
    echo "Error: Virtual environment not found at .venv/bin/activate"
    exit 1
fi

# Get head node address from argument or environment variable
HEAD_NODE_ADDRESS=${1:-${RAY_HEAD_ADDRESS}}

if [ -z "$HEAD_NODE_ADDRESS" ]; then
    echo "Error: Head node address not provided"
    echo "Usage: ./connect.sh <head_node_address>"
    echo "Example: ./connect.sh 192.168.1.100:6379"
    echo "Or set RAY_HEAD_ADDRESS environment variable in .env.${ENV}"
    exit 1
fi

# Set Ray configuration with defaults
RAY_HEAD_PORT=${RAY_HEAD_PORT:-6379}
RAY_NUM_ACTORS=${RAY_NUM_ACTORS:-4}

echo "Ray Worker Configuration:"
echo "  Head Node Address: $HEAD_NODE_ADDRESS"
echo "  Number of Actors: $RAY_NUM_ACTORS"

# Trap for cleanup
cleanup() {
    echo "Stopping Ray worker node..."
    ray stop 2>/dev/null || true
    echo "Worker node stopped"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Start Ray worker node
echo "Connecting to Ray cluster at $HEAD_NODE_ADDRESS..."
ray start --address="$HEAD_NODE_ADDRESS" \
    > ./logs/ray_worker.log 2>&1

if [ $? -eq 0 ]; then
    echo "Successfully connected to Ray cluster!"
    echo "Worker node is running. Logs available at ./logs/ray_worker.log"
    echo "Press Ctrl+C to stop the worker node"
    
    # Keep the script running
    while true; do
        sleep 10
        # Check if Ray is still running
        if ! ray status > /dev/null 2>&1; then
            echo "Ray worker node has stopped unexpectedly"
            exit 1
        fi
    done
else
    echo "Failed to connect to Ray cluster at $HEAD_NODE_ADDRESS"
    echo "Please check that:"
    echo "  1. The head node is running"
    echo "  2. The address is correct (format: host:port)"
    echo "  3. Network connectivity exists between nodes"
    echo "  4. Firewall rules allow connection on port $RAY_HEAD_PORT"
    exit 1
fi