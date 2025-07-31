# NN Trust Applications

This repository collects all works related or based on nn_trust core attack_library




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



