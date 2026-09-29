# NN Trust Applications

Backend and command-line tools for running adversarial-robustness benchmarks with the bundled [
`nn_trust`](submodules/nn_trust) library. A run evaluates one or more models against a dataset, saves per-attack
results, and can generate a PDF report.

## Table of contents

- [Setup](#setup)
    - [CUDA support for `llama-cpp-python`](#cuda-support-for-llama-cpp-python)
- [Run a benchmark](#run-a-benchmark)
- [Supported datasets](#supported-datasets)
- [Supported models](#supported-models)
- [Security report PDF](#security-report-pdf)
- [API](#api)
- [Testing](#testing)
    - [Test files](#test-files)
    - [Test helpers](#test-helpers)

## Setup

Requires Python 3.11, [uv](https://docs.astral.sh/uv/), and Git.

```bash
git submodule update --init --recursive
uv sync --python 3.11
```

### CUDA support for `llama-cpp-python`

The example below uses Ubuntu 24.04, NVIDIA driver 580, CUDA 13.0, and an Ada GPU (compute capability 8.9).
Adjust the toolkit version and GPU architecture for your hardware. The CUDA Toolkit (`nvcc`) is required in addition
to the driver.

Install the toolkit if needed:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install cuda-toolkit-13-0
```

From the project root, configure CUDA and install the package into the project environment:

```bash
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=$CUDA_HOME/bin:$PATH
export CUDAToolkit_ROOT=$CUDA_HOME
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=89" \
  uv pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir
```

Add the exports to `~/.bashrc` to persist them. With the project virtual environment activated, verify GPU support:

```bash
python -c "import llama_cpp; print(llama_cpp.llama_supports_gpu_offload())"   # expected: True
```

Enable GPU offloading when loading a model:

```python
from llama_cpp import Llama

llm = Llama(model_path="model.gguf", n_gpu_layers=-1, verbose=True)
```

Use a smaller positive `n_gpu_layers` value if the model exceeds available VRAM. For missing `nvcc` or `libcudart`
errors, check the toolkit path and environment variables above.

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

## Supported datasets

These are the supported datasets type:

| Type           | Description                                                                                                         | Status    |
|----------------|---------------------------------------------------------------------------------------------------------------------|-----------
| `image_folder` | Classification images organized in one folder per class.                                                            | Supported |
| `flat`         | Images in a single folder, with optional labels in `labels.csv` or `labels.json`.                                   | Supported |
| `parquet`      | Classification images and labels stored in Parquet files, with configurable column names.                           | Supported |
| `coco`         | Object-detection images with annotations in COCO JSON format.                                                       | Supported |
| `prompt_jsonl` | Text/prompt datasets for LLM attacks, stored as JSON/JSONL with a prompt field and optional target/category fields. | Planned   |

## Supported models

These are the supported model type:

| Type            | Description                                                                            |
|-----------------|----------------------------------------------------------------------------------------|
| `plain`         | A complete PyTorch model saved as `model.pth`.                                         |
| `model_weights` | PyTorch weights in `model_state_dict.pth`, with a `Model` class defined in `model.py`. |
| `timm`          | Pretrained image models from the timm library, selected by model ID.                   |
| `torch_script`  | A TorchScript model saved as `model.pt`.                                               |
| `torch_dynamo`  | A model exported with `torch.export`, saved as `model.pt2`.                            |
| `onnx`          | An ONNX model saved as `model.onnx`.                                                   |
| `api`           | A computer-vision model accessed through an API URL.                                   |
| `ultralytics`   | Ultralytics detection models loaded from `model.pt` or a model ID.                     |
| `HuggingFace`   | Hugging Face image-classification or causal language models, selected by task.         |
| `Ollama`        | Language models served by an Ollama instance.                                          |
| `Llamacpp`      | Local GGUF files, including files nested in Hugging Face cache directories.            |
| `Gemini`        | Gemini language models accessed through an API, requiring an API key.                  |
| `OpenRouter`    | Language models accessed through OpenRouter, requiring an API key.                     |

For `Llamacpp`, pass a GGUF file or a directory as `model_path`. Directories are searched recursively;
if multiple distinct GGUF files are found, provide the exact file path to select the model.

## Security report PDF

The default report follows the frontend dashboard on white A4 pages: model information,
preprocessing, metric cards, benchmarking, and risk-based vulnerability assessment.
The generator accepts `ModelReportProps` or a report dictionary. Dictionaries preserve
frontend field order and additional fields such as attack `category`.

Every page displays `report/images/Logo_Leonardo.png` in the top-left header,
within a 140 × 28 pt box preserving its proportions. A minimum 68 pt top margin
keeps content clear of the logo. Pass `header_logo_path` to `generate()` to use
a different logo; the bundled image is used by default, including CLI/API calls.

```python
from report import AdversarialReportGenerator

generator = AdversarialReportGenerator(
    benchmark=[
        {"name": "Reference model", "param": 25_000_000,
         "metrics": {"accuracy": 0.91}},
    ],
    include_attack_details=True,
)
generator.generate(report_data, output_path="out/security-report.pdf")
```

`benchmark` accepts the objects returned by `/report/benchmarks`, a mapping of model
names to those objects, or the legacy metric-to-score-list format. Legacy scores
have no parameter counts, so they appear in the leaderboard only. Accuracy is the
preferred comparison metric; otherwise the first finite scalar metric is selected.
Without supplied benchmarks, the PDF explicitly displays an empty state. The PDF
generator does not fetch benchmarks; existing callers must pass them to enable the
comparison chart.

Individual attack metrics and parameters are included by default, including through
`report.py`. Set `include_attack_details=False` only for a summary-only report.
Metrics whose names start with `Class` (case-insensitive, including `classrobustness`)
are rendered as bar charts in both global metrics and attack details. Lists use
zero-based class IDs on the X axis; dictionaries use their keys as class labels.
Each metric has a single chart containing all classes, even above 24 classes; X-axis
tick labels are hidden. Missing or non-finite values are marked `N/A`, not plotted
as zero.

Confusion matrices are shown as heatmaps using the original values, both globally
and for individual attacks. Each attack starts on a new page; parameters use compact
cards, preserving their values without metric rounding. Long metric arrays use
splittable rows so they can continue across pages. Input data is never modified.

Use `AdversarialReportStyle(pagesize=landscape(A4))` for landscape output. Both
orientations stack the chart and leaderboard to allow long rankings to paginate.

Examples are loaded from `<report repository>/<attack ID>/log.pth` (tensor-only
loading) or legacy `<sample>_original.png`, `_pert.png`, `_adv.png` images (JPEG is
also supported). The top-level `repository` provided by `repository_router.py` is
used automatically. For other callers, pass `examples_root` to `generate()`; the
CLI uses the directory containing `report.json`. Tensor images are denormalized
with the report's preprocessing mean/std when compatible; perturbations are shown
as absolute differences scaled for visibility. Missing examples are indicated in
the report. Original artifacts and metrics are never changed.

Run the focused PDF tests with:

```bash
uv run pytest test/test_report_pdf.py -q
```

## API

Start the FastAPI server:

```bash
uv run python app.py --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` for the interactive API. The main benchmark endpoints are `POST /job/start_benchmark`,
`GET /job/getJobs`, and `GET /job/getReport`. Server paths and other settings can be supplied as CLI options or in a
JSON file passed with `--configuration_file`.

For a complete API request example and CIFAR-10 helper commands, see [benchmarking/README.md](benchmarking/README.md).

## Testing

Application tests are in [`test/`](test). Run the full suite from the project root:

```bash
uv run python -m pytest test -ra
```

Device-parametrized tests run on CPU and also CUDA when available. Execution and single-attack tests use pretrained
ResNet-18 weights and cached dog images, downloading them when missing. Dataset and router tests create small temporary
datasets; router tests also create a temporary model and do not need an external benchmark request JSON file.
The bundled library has its own tests in [`submodules/nn_trust/tests/`](submodules/nn_trust/tests), which are not
included
in the command above.

To save full failure tracebacks and preserve pytest's exit status in Bash:

```bash
set -o pipefail
uv run python -m pytest test -ra --tb=long 2>&1 | tee test_results.txt
```

The verified run on September 22, 2026 completed with **87 passed, 0 failures, and 2 deprecation warnings** in 23.32
seconds,
including CPU and CUDA cases. The warnings concern Pydantic class-based configuration and Starlette TestClient's use of
httpx. See the [error and fix summary](test_error_summary.txt), [original tracebacks](test_errors_initial.txt),
[intermediate verification logs](test_errors_recheck.txt), and [final results](test_results_final.txt).
The suite uses the current serial executor interface; obsolete Ray variants were removed and a regression test for
ground-truth attack targets was added.

Run the dataset loading and transformation tests with:

```bash
uv run pytest test/test_dataset_loader.py -q
```

These tests create small temporary datasets and require no downloads or local dataset repository. The COCO test is
skipped when `pycocotools` is unavailable. Each test includes a short description of the behavior it checks.

### Test files

| File                                                        | Description                                                                                                                     |
|-------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------|
| [test_dataset_loader.py](test/test_dataset_loader.py)       | Image-folder, flat, Parquet, and COCO loading; format validation, subsets, preprocessing, inverse transforms, and crop padding. |
| [test_evaluation.py](test/test_evaluation.py)               | Classification, detection, and AdvYOLO evaluation; ground-truth targets, artifacts, progress, errors, and logger cleanup.       |
| [test_execution.py](test/test_execution.py)                 | Local attack execution, saved artifacts, and benchmark metrics on available devices.                                            |
| [test_job_router.py](test/test_job_router.py)               | Benchmark API creation, scheduling, job listing, reports, and errors, using temporary model and dataset repositories.           |
| [test_job_tracking.py](test/test_job_tracking.py)           | Baseline attack status and intermediate batch progress.                                                                         |
| [test_logger.py](test/test_logger.py)                       | Checkpoint artifact logging and per-tag limits.                                                                                 |
| [test_model_info.py](test/test_model_info.py)               | Preprocessing size inferred from input dimensions, including precedence over an explicit transformation size.                   |
| [test_parameter_utils.py](test/test_parameter_utils.py)     | Preservation of zero defaults in parameter metadata.                                                                            |
| [test_parquet_streaming.py](test/test_parquet_streaming.py) | Restartable, bounded Parquet streaming and worker partitioning.                                                                 |
| [test_report_pdf.py](test/test_report_pdf.py)               | PDF formatting, pagination, charts, benchmarks, and saved examples.                                                             |
| [test_report_router.py](test/test_report_router.py)         | Scalar metric filtering in benchmark API responses.                                                                             |
| [test_single_attack.py](test/test_single_attack.py)         | Single-image classification attacks, confidence for each iteration, and sanitization of image output.                           |

### Test helpers

| File                                        | Description                                               |
|---------------------------------------------|-----------------------------------------------------------|
| [utils/devices.py](test/utils/devices.py)   | Lists available CPU and CUDA test devices.                |
| [utils/utils.py](test/utils/utils.py)       | Provides cached sample images, a model, and a dataloader. |
| [utils/__init__.py](test/utils/__init__.py) | Exports shared test helpers.                              |
| [__init__.py](test/__init__.py)             | Marks the test directory as a Python package.             |
