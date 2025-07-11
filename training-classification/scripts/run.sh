echo "Starting training procedure"

tensorboard --bind_all --port 6006 --logdir train_out &

python /app/train.py \
    -d /app/data/clean-aircraft-crop-few-birds \
    -o /app/train_out \
    -m train \
    --batch 256 \
    --lr 6e-3 \
    --only_classifier \
    --temperature 3.0 \
    --label_smoothing 0.1 \
    --weight_decay 0.005 \
    --model_size large \
    --warmup 10
