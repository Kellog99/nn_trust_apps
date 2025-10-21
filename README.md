# NN Trust Applications

This repository collects all works related or based on nn_trust core attack_library 

## 🚀 Featured Apps

1. [**TITANN Server**](./attack-server)  
   A framework to manage incoming frontend request and execute and manage jobs realted to other apps (e.g.: benchmarking, single image attacks, etc)

2. [**Benchmarking**](./benchmarking)  
   An application of `nn_trust` that has the goal of benchmarking a target model performance, over a selected dataset using SoTA AML techniques

3. [**Single Image attack Demo**](./image-attack)  
   Its a Gradio based demo aiming at displaying the process of executing an Adversarial attack on a single image on a target model. Displaying also all intermediate steps and metrics.

4. [**Classification Model Training**](./training-classification)  
   A minimal set of tools to traing a vision classification model. This is the code used to train Aircraft based model used in the '25 Summer Demo.



## Usage

Each folder contains an application requiring as a git submodule `nn_trust`.      

To correctly install the submodule, remember to initialize the submodule with the following commands
```bash
git submodule init && git submodule update
```

Each application specifies its dependencies in its local `pyproject.toml` file. 
A new Python virtual environment may be initialized using `uv` by running in the respective application folder `uv sync`.

### Examples - Add a submodule targeting a specific branch

`git submodule add -b main git@github.com:LeoPhilosophers/nn_trust.git`

### Example - Update a submodule branch configuration

`git config -f .gitmodules submodule.attack-server/submodules/nn_trust.branch develop
`

### How to install `nn_trust` as a  submodule

To install `nn_trust` in your Python virtual environment, when in an application folder (e.g. `image-attack`), use 
```bash
pip install -e submodules/nn_trust
```
or 
```bash
uv pip install -e submodules/nn_trust
```
if using `uv` to manage the Python virtual environment.



