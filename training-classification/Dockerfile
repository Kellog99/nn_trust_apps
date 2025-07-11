FROM python:3.12-slim


# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libjpeg-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip install -r requirements.txt

ENV TORCH_HOME=/app/torch_cache

COPY finetune_resnet50.py /app/train.py
