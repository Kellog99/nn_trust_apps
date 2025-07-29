# docker run -it --gpus all -p 7860:7860 nndemo bash ./assets/scripts/run_one_iamge.sh
echo "STARTING APP..."

uv run main.py --labels_path ./assets/data/birds_plane_classes.json