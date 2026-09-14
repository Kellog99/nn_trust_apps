# NN Trust Applications

Applications built on [`nn_trust`](submodules/nn_trust) for adversarial-robustness evaluation, benchmarking, and report
generation. The project exposes both a CLI and a FastAPI backend.

## Features

These are all the engineering that has been done.

- Execution:
    - Local
    - Ray-basedl
- Supported **models**:
    - Computer-vision
    - NLP model
- Supported **datasets**:
    - Image-folder
    - flat-image
    - Parquet
- Attack/metric selection and per-attack progress tracking
- JSON benchmark results and PDF report generation
- Repository discovery and model/dataset upload endpoints

## Requirements and installation

- Python 3.11
- [`uv`](https://docs.astral.sh/uv/)
- Git with submodule support

```bash
uv sync --python 3.11
git submodule update --init --recursive
uv pip install -e submodules/nn_trust/
```

## Quick start

Start the API:

```bash
python app.py --reload --host 0.0.0.0 --port 8000
```

The interactive API documentation is available at `http://localhost:8000/docs`.

Run a benchmark from a YAML configuration file:

```bash
python benchmark.py --config_path path/to/config.yaml
```

Or use `POST /job/start_benchmark`. The API accepts `ModelInfo`, `DatasetInfo`, attack and metric selections, and
benchmark options. `GET /job/getJobs` returns attack status, progress, and errors for a benchmark.

## Dataset handling

Dataset loading is selected explicitly with `DatasetInfo.dataset_type`:

| Type           | Expected layout                                                                                                             |
|----------------|-----------------------------------------------------------------------------------------------------------------------------|
| `image_folder` | `data/<class_name>/<image>` (or the configured `folder_data` split)                                                         |
| `flat`         | Images directly in one directory; optional `labels.csv` (`file,label`) or `labels.json` mapping filenames to integer labels |
| `parquet`      | One `.parquet` file or a directory of Parquet shards                                                                        |

Parquet datasets are read as restartable, bounded-memory streams. Configure their columns in `DatasetInfo.parquet_info`:

```json
{
  "dataset_type": "parquet",
  "parquet_info": {
    "image_column": "image",
    "image_key": "bytes",
    "label_column": "label"
  }
}
```

`get_dataloader()` accepts either `dataset_path` or `dataset_info.repository`, applies the model transformation,
supports worker partitioning, and honors the benchmark `subset`. Unsupported formats such as COCO, YOLO, video, or
medical volumes should be supplied as a compatible `torch.utils.data.Dataset` rather than inferred from a path.

## Model handling

`ModelInfo.model_type` selects the loader. Available values are:

`plain`, `timm`, `torch_script`, `torch_dynamo`, `onnx`, `api`, `HuggingFace`, `Ollama`, `Gemini`, and `OpenRouter`.

Common local layouts are:

```text
model_repository/<model-id>/
├── info.json
└── model.pth                 # plain PyTorch module
```

The other local formats expect `model.pt` (TorchScript), `model.pt2` (TorchDynamo export), `model.onnx` (ONNX), or
`model_state_dict.pth` plus `model.py` (model definition and weights). `timm` loads a pretrained model by its timm ID;
Hugging Face can load either a computer-vision checkpoint or a language model.

Remote NLP models use their model ID and the corresponding credentials: `GEMINI_API_KEY` or `OPENROUTER_API_KEY` for
those providers. Ollama defaults to `http://localhost:11434`; custom endpoints are provided through the model API field.

All models are wrapped in the shared `nn_trust` adapter interface, moved to the selected device, and put in evaluation
mode before benchmarking.

## Metadata and repositories

Each model and dataset must have an `info.json`. At minimum, provide `id`, `name`, `task`, and `input_dimensionality`;
benchmark execution also requires a valid `repository` path. Dataset metadata additionally supports `batch_size`,
`num_workers`, `dataset_type`, `folder_data`, and `parquet_info`. Model metadata supports `model_type`,
`transformation`, and `api` where applicable.

Resources are typically stored as:

```text
model_repository/<model-id>/info.json
dataset_repository/<dataset-id>/info.json
benchmark_repository/<benchmark-id>/<model-id>/<dataset-id>/report.json
```

The repository API can list resources by type and task. Models can be uploaded as ZIP packages containing exactly one
supported model file (`.pt`, `.pth`, `.pkl`, or `.pickle`) and one JSON metadata file; dataset uploads accept ZIP
archives.

## Benchmark output

Each run creates a unique benchmark ID and stores a `report.json` containing model information, requested metrics,
attack results, and saved examples. The identity baseline is added automatically when it is not selected, providing the
unperturbed performance reference.

To generate a PDF from benchmark output:

```bash
python report.py \
  --benchmark_path path/to/report.json \
  --output_path path/to/reports
```

## Repository layout

```text
attack_server/       # API/job services
benchmarking/        # benchmark execution and evaluation
models/              # Pydantic metadata and configuration models
utils/dataset/       # dataset loaders
utils/model/         # model loaders
report/              # PDF report generation
submodules/nn_trust/ # attack and adapter library
```
