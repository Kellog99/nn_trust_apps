docker build -t train_aircraft .

docker run -it --rm \
    --gpus all \
    --ipc=host \
    -p 6006:6006 \
    -v "$(pwd)/data:/app/data" \
    -v "$(pwd)/torch_cache:/app/torch_cache" \
    -v "$(pwd)/scripts:/app/scripts" \
    -v "$(pwd)/train_out:/app/train_out" \
    train_aircraft bash /app/scripts/run.sh

