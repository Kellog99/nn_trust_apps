# NN Trust Applications

## Table of contents:

1. **Intro**
2. **Prerequisites**
3. **Execution**
5. **Development notes and layout**

### 1. Intro

This repository represents a collection of applications and tooling that build on top of the `nn_trust` library for
adversarial machine learning (attacks, benchmarks, and demos).

This repository hosts several related projects in one workspace. The most important subprojects are:

- `attack_server/` — TITANN backend and job manager (FastAPI + Ray + Celery integration)
- `benchmarking/` — benchmark runner and utilities that orchestrate model evaluations and aggregate results
- `image-attack/` — Single-image demo (Gradio) showcasing an adversarial attack flow
- `training-classification/` — utilities and examples for training classification models used in demos
- `report/` — Creates a PDF report from a json file associate with a benchmark

### 2. Prerequisites

Here are all the step for using this repository:

1. **Pakages**: since all the packages are handled by `uv` (https://docs.astral.sh/uv/); hence to create a fully working
   environment do:
    ```bash
      uv sync --python 3.11
    ```
2. **Submodules**: if they are *not present* then this is the command for using the `submodules`:
    * **Downloading**: download the submodules in the corresponding folder `./submodules/name` from `git`:

        ```bash
        git submodule add git@github.com:LeoPhilosophers/nn_trust.git submodules/nn_trust
        git submodule add git@github.com:LeoPhilosophers/data-quality.git submodules/data_quality
        ```
    * **Initialization**: to initialize the submodules execute
      ```bash
      git submodule init
      git submodule update --recursive
      ```
3. **Installation**: install locally `nn_trust` dependency (from submodule)

    ```bash
    # create & activate a venv, then
    pip install -e submodules/nn_trust/
    pip install -r requirements.txt  # or use the subproject's pyproject / uv workflow
    ```

### 3. Execution

Now it is possible to execute all the functionalities of the STABLE-AI framework. Here there are all the commands:

1. **Application**: this part is for using the `GUI`. To do so it is necessary to start the FastAPI server inside:

    ```bash
    python app.py --reload --host 0.0.0.0 --port 8000
    ```
2. **Benchmarking**: this command is for executing just the benchmarking on a specific `dataset-model`:
    ```bash
   python benchmark.py --MODELPATH path/to/model --DATASETPATH path/to/dataset --CONFIGPATH path/to/config
    ```
   The *configuration path* handles all the attacks' configuration. This avoid to pass all the arguments that could be a
   lot through terminal.

3. **Report**: to produce a report regarding the benchmark of a specific model:
    ```bash
   python report.py --OUTPUTDIR path/to/output_folder
    ```
   The *OUTPUTDIR* represents the path to the benchmark's output folder. In this folder there are all the information
   for generating the pdf report.

### 5. Development notes & repository layout

- `attack_server/` — FastAPI app and routers. Key files:
    - `attack_server/app.py` — FastAPI application entry
    - `attack_server/routers/job_router.py` — job endpoints, Ray executor integration
    - `attack_server/lib/disk_reader.py` — helpers for locating benchmark outputs

- `benchmarking/` — benchmark runner and utilities. Key files:
    - `benchmarking/main.py` — postprocessing and runner entrypoints
    - `benchmarking/benchmark_utils/` — executor, evaluator, and helpers

- `image-attack/` — demo UI and static assets.

If you edit Python code, prefer editing the module inside the corresponding subfolder and run the local unit tests when
available. The `nn_trust` submodule contains the core attack implementations.

