# Image Attack

To run this app, launch
```
uv sync && source .venv/bin/activate && uv pip install submodules/nn_trust && uv run -s main.py
```
To use a specific model, load the model inside `assets/model`. 
To use a specific label names, use `--labels_path=path/to/labels.json`.


To build the container is defined in `Dockerfile` with the building target `generic-demo`. 

### Image building 
Before builiding: install the requirements for [nvidia container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), in case `cuda` device might be used.
From within the project directory execute
```bash
docker build -t nntrust-generic-demo . 
```
This command will create a docker image with name `nntrust-generic-demo` of the local repository `nn_trust` with a pre-defined local model as specified in `MODEL_PATH`. You can check using `docker image ls` or `docker image list` to see whether the image has been built correctly.

### Runinng the demo
Either Docker or Podman works fine with the building process, however it is required Podman with version greater than 5, in order to have localhost ports to be effectively routed back.
To execute correctly the demo in `example/one_image_attacks.py`, execute 
```bash
docker run -p 7860:7860 -it nntrust-generic-demo:latest uv run -s --no-default-groups example/one_image_attacks.py
```

# Running Demo
## With Docker
In order to use gpu through docker make sure to install [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
```
docker run -it --gpus all -p 7860:7860 nndemo
```
