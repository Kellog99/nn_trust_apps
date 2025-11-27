# Getting Started: Attack Server

This part represents the collection of all the services that are needed for the Front End to run.

* Dependencies
* Environment Configuration
* Repositories Structure

## Dependencies

First of all, it is necessary to check the dependencies that are required for running the application. To do that it is
necessary to be positioned at the top level directory of `attack-server`:

### Installation

The core package for this is `nn_trust`. Hence, it is necessary to check its correct versioning.

#### nn_trust version

To check the `nn_trust` version is necessary to go to the its folder in the submodules and check whether the branch is
the correct one.

```bash
cd attack-server/submodules/nn_trust #same for benchmarking submodule!
git status #see the branch of the package
git checkout <desired_branch>
git pull origin <desired_branch>
```

A stable branch to work with is the `develop`.

#### Python packages

Once this has been done it is possible to install every dependency needed for this application to run.

```bash
uv pip install -e submodules/nn_trust
uv pip install timm   #somehow needed outside the lock
uv sync
```

## Environment Configuration

The following guide is meant to walk you through the environment variables necessary to set up the server.

### System Configuration

Core FASTAPI application settings:

```bash
ENVIRONMENT=dev
DEBUG=true
LOG_LEVEL=info
HOST=0.0.0.0
PORT=8000
WORKERS=1
```

### Repository

The repositories play an important role for the application because they collect all the results that have been produced
over the time. There are mainly 3 types of repositories:

1. Model Repository
2. Dataset Repository
3. Report Repository

Hence, it is important to define their path:

```bash
MODEL_REPO=/path/to/your/model_repo
DATASETS_REPO=/path/to/your/datasets
```

#### Model repository

This contains all the models that have been given to the application. The repository is a folder of folders whose
contain the actual model and its information. The structure is given by

```
model_repository/
├── model_1/                        # Folder containing the model 1
    ├── model_1.pth                 # actual model
    ├── model_1.json                # Json containing the model's information
└── ...                             
```

## Dataset and Model Structure

### Dataset Repository Structure

Datasets must follow the data quality structure format:

```
dataset_folder_name/
├── data/                           # Folder containing dataset files
└── dataset_folder_name.json        # Metadata file
```

### Dataset Metadata Example

The metadata JSON file must include the following fields:

```json
{
  "name": "imagenet_subset",
  "description": "ImageNet validation subset for quick testing",
  "num_classes": 1000,
  "subset": 5000,
  "batch": 32,
  "label_dict": {
    "0": "tench",
    "1": "goldfish",
    "2": "great_white_shark",
    "3": "tiger_shark"
  },
  "type_dataset": 2,
  "type": "image",
  "mode": "classification",
  "num_workers": 4,
  "source_path": "/home/user/datasets/imagenet_subset/data",
  "transform_config": {
    "size": 256,
    "crop": 224,
    "transform_id": "imagenet_like_crop",
    "mean": [
      0.485,
      0.456,
      0.406
    ],
    "std": [
      0.229,
      0.224,
      0.225
    ]
  }
}
```

## Model Upload Structure

To upload a model, compress the following two files into a ZIP archive:

```
model.zip
├── model.pth     # PyTorch model (architecture + weights)
└── model.json    # Model metadata
```

**Important**: Both files must have the same base name with different extensions.

### Model Metadata Example

```json
{
  "name": "resnet50_imagenet",
  "pretrained": true,
  "num_classes": 1000,
  "task": "classification"
}
```

### Model Metadata Fields

- **`name`**: Model identifier
- **`pretrained`**: Whether the model uses pretrained weights
- **`num_classes`**: Number of output classes
- **`task`**: Model task type (e.g., `classification`, `detection`)

### Model Upload Limits & TIMM's pre-loaded desired models file

Inside timm_models.json, the user must configure the timm models that the API will be able to use.

```bash
TIMM_MODELS_JSON_PATH="resources/timm_models.json"
MAX_MODEL_SIZE_UPLOAD=5000
MAX_MODEL_JSON_SIZE_UPLOAD=5000
```

### Ray Cluster Configuration

Configure the distributed computing cluster head:

```bash
RAY_HEAD_HOST=127.0.0.1
RAY_HEAD_PORT=6379
RAY_DASHBOARD_PORT=8265
RAY_PY_MODULES=/path/to/nn_trust_apps/benchmarking   
```

**For worker nodes**, uncomment and configure:

```bash
# RAY_HEAD_ADDRESS=192.168.1.100:6379
```

#### Ray Actor Configuration

Control resource allocation for parallel execution:

- **`RAY_NUM_ACTORS`**: Number of parallel Ray actors (default: `1`)
- **`FRACTION_FOR_GPU_ACTOR`**: GPU fraction per actor (default: `1` = full GPU per actor)

```bash
RAY_NUM_ACTORS=1
FRACTION_FOR_GPU_ACTOR=1
```

> **Note**: Adjust these values based on your hardware. For example, with 2 GPUs and `RAY_NUM_ACTORS=4`, set
`FRACTION_FOR_GPU_ACTOR=0.5` to allocate 2 actors per GPU.

### Benchmark Configuration

Settings specific to benchmark execution:

```bash
BENCHMARK_LOAD_RESULTS=False
BENCHMARK_OVERWRITE=True
BENCHMARK_NUM_IMAGES_TO_SAVE=-1
BENCHMARK_SAVE_PERTURBATION=False
BENCHMARK_GPU=True
BENCHMARK_OUTPUT_DIR=/path/to/output
BENCHMARK_OUTPUT_FORMAT=report
```

## Launch Application

### Using the Startup Script

The simplest way to launch the entire application stack:

```bash
./start_env.sh
```

This script will automatically start all required services based on your `.env.development` configuration.

Check Ray dashboard at `http://HOST:8265` for cluster health
