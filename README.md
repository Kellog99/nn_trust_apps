# NN Trust Applications

This repository collects all works related or based on nn_trust core attack_library

# NN Trust Applications

A collection of applications and tooling that build on top of the `nn_trust` library for adversarial machine learning (
attacks, benchmarks, and demos).

This repository hosts several related projects in one workspace. The most important subprojects are:

- `attack-server/` — TITANN REST backend and job manager (FastAPI + Ray + Celery integration)
- `benchmarking/` — benchmark runner and utilities that orchestrate model evaluations and aggregate results
- `image-attack/` — Single-image demo (Gradio) showcasing an adversarial attack flow
- `training-classification/` — utilities and examples for training classification models used in demos
- `report/` — Creates a PDF report from a json file associate with a benchmark

Contents of this README

- Quick start (prereqs + run attack-server)
- Running benchmarks
- Single-image demo
- Development notes and layout
- Important environment variables

## Quick start

Prerequisites

- Python 3.10+ (check individual `pyproject.toml` files in subfolders for specific requirements)
- Git with submodules enabled
- (Optional) Docker for containerized runs

Initialize submodules

```bash
git submodule init
git submodule update --recursive
```

Install local `nn_trust` dependency (from submodule)

From a subproject folder (for example `attack-server/`):

```bash
# create & activate a venv, then
pip install -e submodules/nn_trust/
pip install -r requirements.txt  # or use the subproject's pyproject / uv workflow
```

Run the TITANN attack server (development)

1. Set environment variables used by the server. A minimal example:

```bash
export BENCHMARK_OUTPUT_DIR="/path/to/benchmark_out"
export INTERNAL_MODEL_STORAGE="/path/to/model_metadata"
export INTERNAL_DS_STORAGE="/path/to/dataset_metadata"
export RAY_NUM_ACTORS=1
# other env vars are read from the attack_server code (see Important env vars section)
```

2. Start the FastAPI server inside `attack-server/`:

```bash
# from repository root
cd attack_server
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

The server exposes endpoints under `/job` (see `attack-server/routers/job_router.py`).

## Running benchmarks

The `benchmarking/` package contains the orchestration logic for running evaluations on models and datasets.

- To run a benchmark programmatically, the attack server calls into `benchmarking.benchmark()` which drives a Ray-based
  executor.
- Output is written under the configured `BENCHMARK_OUTPUT_DIR` (default `./benchmark_out`). Each benchmark run creates
  a timestamped task folder with per-model `aggregate*.json` and `info.json` files.

If you run `benchmarking` directly, use the `pyproject.toml` and its CLI (see `benchmarking/main.py`).

## Single-image demo (image-attack)

Open `image-attack/` for a Gradio demonstration of single image attacks. Example usage:

```bash
cd image-attack
pip install -r requirements.txt
python main.py
# or run the provided demo container script
./run_demo.sh
```

## Development notes & repository layout

- `attack-server/` — FastAPI app and routers. Key files:
    - `attack-server/app.py` — FastAPI application entry
    - `attack-server/routers/job_router.py` — job endpoints, Ray executor integration
    - `attack-server/lib/disk_reader.py` — helpers for locating benchmark outputs

- `benchmarking/` — benchmark runner and utilities. Key files:
    - `benchmarking/main.py` — postprocessing and runner entrypoints
    - `benchmarking/benchmark_utils/` — executor, evaluator, and helpers

- `image-attack/` — demo UI and static assets.

If you edit Python code, prefer editing the module inside the corresponding subfolder and run the local unit tests when
available. The `nn_trust` submodule contains the core attack implementations.

## Important environment variables

- `BENCHMARK_OUTPUT_DIR` — where benchmark runs are written (default `./benchmark_out`)
- `INTERNAL_MODEL_STORAGE` — directory holding model metadata JSON and weights
- `INTERNAL_DS_STORAGE` — directory holding dataset metadata JSON
- `RAY_NUM_ACTORS` — number of actors used by Ray executor
- `RAY_PY_MODULES` — optional Python modules path for Ray runtime_env

There are additional flags used by various scripts; search for `os.environ.get(` in the subpackages to see the full
list.

## Troubleshooting

- If endpoints complain about missing files under `BENCHMARK_OUTPUT_DIR`, confirm the benchmarking run completed and
  that `info.json` / `aggregate*.json` files exist under the model folder.
- If Ray initialization fails, ensure a compatible Ray version is installed and that the environment variables are
  correct.

## Contributing

1. Fork the repo and create a branch for your feature/fix.
2. Keep changes scoped to the subproject when possible (e.g., only edit `attack-server/` for web/API changes).
3. Run available tests in `submodules/nn_trust/tests` and local tests in subprojects.

## Contact / Where to look next

- API routes: `attack-server/routers/*.py`
- Benchmark output handling: `attack-server/lib/disk_reader.py` and `benchmarking/main.py`
- Ray executor: `benchmarking/benchmark_utils/executor.py`

If you'd like, I can also add quick-start scripts, or a sample `.env` file for `attack-server` listing the minimal env
vars.



