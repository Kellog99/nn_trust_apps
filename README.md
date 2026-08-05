# NN Trust Applications

A collection of applications built on top of **`nn_trust`** for adversarial machine learning, robustness evaluation, benchmarking, and report generation.

The repository provides both command-line utilities and backend services for evaluating machine learning models against adversarial attacks and producing reproducible benchmark reports.

---

# Features

- FastAPI backend for attack execution
- Benchmark orchestration
- Adversarial attack evaluation
- PDF report generation
- Repository management for models, datasets, and reports
- Integration with Ray for distributed execution

---

# Repository Layout

```text
.
├── attack_server/      # FastAPI backend and job manager
├── benchmarking/       # Benchmark runner and utilities
├── report/             # PDF report generation
├── submodules/
│   ├── nn_trust/
│   └── data_quality/
└── ...
```

Main components:

| Directory | Description |
|-----------|-------------|
| `attack_server/` | FastAPI backend, Ray integration and job management |
| `benchmarking/` | Benchmark execution and evaluation utilities |
| `report/` | Generates PDF reports from benchmark outputs |
| `submodules/nn_trust` | Core adversarial attack library |

---

# Requirements

- Python **3.11**
- [`uv`](https://docs.astral.sh/uv/)
- Git with submodule support

---

# Installation

## 1. Create the environment

```bash
uv sync --python 3.11
```

## 2. Clone the submodules

If the submodules are not already available:

```bash
git submodule add https://github.com/Kellog99/nn_trust.git submodules/nn_trust
git submodule add https://github.com/Kellog99/data_quality.git submodules/data_quality
```

Initialize them:

```bash
git submodule init
git submodule update --recursive
```

## 3. Install `nn_trust`

```bash
uv pip install -e submodules/nn_trust/
```

---

# Quick Start

## Launch the backend

```bash
python app.py --reload --host 0.0.0.0 --port 8000
```

---

## Run a benchmark

```bash
python benchmark.py \
    --model_path path/to/model/info.json \
    --dataset_path path/to/dataset/info.json
```

---

## Generate a report

```bash
python report_class.py \
    --OUTPUTDIR path/to/output_folder
```

The output directory must contain the benchmark results produced by the benchmark runner.

---

# Benchmark CLI

Display the complete list of available options:

```bash
python benchmark.py --help
```

### Main arguments

| Argument | Description |
|----------|-------------|
| `--model_path` | Path to the model `info.json` |
| `--dataset_path` | Path to the dataset `info.json` |
| `--attacks` | List of attacks to execute |
| `--metrics` | Metrics to compute |
| `--output_path` | Directory where benchmark results are stored |
| `--use_ray` | Enable distributed execution with Ray |

---

# Repository Organization

The framework relies on three repositories.

## Model Repository

```text
model_repository/
├── model_1/
│   ├── model.pth
│   └── info.json
└── model_2/
```

---

## Dataset Repository

The dataset repository follows the same organization.

```text
dataset_repository/
├── dataset_1/
│   ├── data/
│   └── info.json
└── dataset_2/
```

---

## Report Repository

```text
report_repository/
└── run_id/
    ├── dataset_1/
    │   ├── model_1/
    │   │   ├── report.json
    │   │   └── examples/
    │   └── model_2/
    └── dataset_2/
```

---

# Metadata (`info.json`)

Every model and dataset is described by an `info.json` metadata file.

Typical information includes:

- Identifier
- Name
- Task
- Domain
- Input dimensionality
- Description
- Repository information
- Number of classes
- Dataset- or model-specific fields

These metadata files are mandatory and are used by the framework to correctly load resources.

---

# Benchmark Output

Each benchmark execution produces a JSON report containing:

- Model information
- Performance metrics
- Robustness metrics
- Attack statistics
- Confusion matrices
- Attack-specific measurements

The generated JSON file is used as input for the report generator.

---

# Development

Relevant modules:

```text
attack_server/
    app.py
    routers/
    lib/

benchmarking/
    main.py
    benchmark_utils/

report/
```

The actual adversarial attack implementations are located inside the `submodules/nn_trust` repository.

---

# Working with Git Submodules

To safely remove a submodule:

```bash
git submodule deinit -f path/to/submodule
git rm -f path/to/submodule
rm -rf .git/modules/path/to/submodule
```

---

# Notes

- Dependencies are managed with **uv**.
- Every model and dataset must provide an `info.json` file.
- Ray is optional and can be enabled with the `--use_ray` flag.
- The report generator expects the output of a completed benchmark execution.