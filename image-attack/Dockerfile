# A docker image that creates an environment to run the demo.
#
# It uses python:3.12.3-slim and uv version 0.7.17.
#
# **Note**: note that although the version of pytorch is built with cuda, 
# the environment might not load the cuda gpu driver.
#
# **Note**: Before building it is necessary to have a pytorch model in pth format that can
# be loaded directly via `torch.load(MODEL_PATH, weights_only=False)`.
# For example, one could use the following script to automatically load a torchvision model: 
# ```python
# import torch
# import torchvision
# 
# model = torchvision.models.resnet101(weights=torchvision.models.ResNet101_Weights.DEFAULT)
# torch.save(model, "model.pth")
# ```
#
# **Note**: To run it via cuda, it needs nvidia container toolkit, the device available on the machine
# are then listed via `nvidia-ctk cdi list`
#
# To run and build the docker container follow the instructions:
# > docker build -t nntrust-generic-demo --build-arg MODEL_PATH=path/to/the/model.pth --target generic-demo .
# To check everything works fine
# > docker run -it nntrust-generic-demo:latest /bin/bash
# To actually host the demo: 
# > docker run -it --device nvidia.com/gpu:0 -p 7860:7860 nntrust-generic-demo:latest uv run --no-default-groups -s example/one_image_attacks.py

# Use a slim python image as building block
FROM python:3.12.3-slim

# Install uv and update PATH
ADD https://astral.sh/uv/0.7.17/install.sh /uv-installer.sh

SHELL ["/bin/bash", "-c"]


# gradio expose ports and server name
EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"

# Define the path of the model checkpoint to copy
ARG MODEL_PATH

# Set the working directory
WORKDIR /app/demo

# Copy the current directory into the container


# Install curl 
RUN apt update && \
    apt install -y curl && \ 
    sh /uv-installer.sh && \ 
    rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

# Use uv to manage all dependencies based on pyproject.toml
COPY pyproject.toml /app/demo
COPY ./submodules /app/demo/submodules
RUN uv sync
RUN uv pip install ./submodules/nn_trust

COPY ./style.css /app/demo
COPY ./titann-logo.png /app/demo
COPY ./assets /app/demo/assets
COPY ./blocks /app/demo/blocks
COPY ./main.py /app/demo

#RUN uv pip install --system -r pyproject.toml --group demo

#RUN uv pip install --system .

#RUN uv sync --system --group demo

CMD ["uv", "run", "main.py"]