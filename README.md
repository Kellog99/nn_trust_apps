# NN Trust Applications

Backend and command-line tools for running adversarial-robustness benchmarks with the bundled [`nn_trust`](submodules/nn_trust) library. A run evaluates one or more models against a dataset, saves per-attack results, and can generate a PDF report.

## Setup

Requires Python 3.11, [uv](https://docs.astral.sh/uv/), and Git.

```bash
git submodule update --init --recursive
uv sync --python 3.11
```

## Run a benchmark

Models and datasets are local directories with an `info.json` file. Point the CLI configuration at those directories:

```yaml
model:
  - source_path: /path/to/model
datasets:
  - source_path: /path/to/dataset
attacks:
  - id: fgsm
metrics:
  - id: accuracy
options:
  output_path: ./benchmark_out
  subset: 100
  gpu: false
```

Run it with:

```bash
uv run python benchmark.py --config_path path/to/config.yaml
```

The identity baseline is always included. Results are written to:

```text
<output_path>/<benchmark_id>/<model_id>/<dataset_id>/
├── report.json
└── <attack_id>/job_results.json
```

Create a PDF from a completed report:

```bash
uv run python report.py --benchmark_path path/to/report.json --output_path ./reports
```

## API

Start the FastAPI server:

```bash
uv run python app.py --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` for the interactive API. The main benchmark endpoints are `POST /job/start_benchmark`, `GET /job/getJobs`, and `GET /job/getReport`. Server paths and other settings can be supplied as CLI options or in a JSON file passed with `--configuration_file`.

For a complete API request example and CIFAR-10 helper commands, see [benchmarking/README.md](benchmarking/README.md).
