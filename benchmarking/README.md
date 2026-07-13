# Benchmarking

In this part it will be explained how to perform a `benchmark` for a specific model.

The benchmark expects models and datasets to be available locally.

For CIFAR-10 experiments, the following helper scripts can be used:

```bash
uv run python scripts/download_cifar10_test_dataset.py
uv run python scripts/download_cifar10_resnet20.py
uv run python scripts/download_cifar10_resnet32.py
```

## Running the benchmark through the API


Start the FastAPI server from the repository root:

If the local CIFAR models were saved from `pytorch-cifar-models`, the original module must be importable when loading the model. In that case, include the Torch Hub cache path:

```bash
PYTHONPATH=.:submodules/nn_trust:${HOME}/.cache/torch/hub/chenyaofo_pytorch-cifar-models_master \
uv run python app.py --host 127.0.0.1 --port 8000
```

Then open the API documentation at:

```text
http://127.0.0.1:8000/docs
```

and use:

```text
POST /job/start_benchmark
```

The request body contains a default configuration file generated from the `BenchmarkExecutionConfig` schema defined in `models/benchmark.py`.


### Configuration File

The default configuration file, which can be edited directly in the request body, is defined below:

```json
{
  "model": {
    "id": "resnet20",
    "name": "resnet20",
    "task": "classification",
    "domain": "computer_vision",
    "num_classes": 10,
    "input_dimensionality": [3, 32, 32],
    "model_type": "plain",
    "repository": "benchmark_assets/models/cifar10_resnet20",
    "transformation": {
      "mean": [0.4914, 0.4822, 0.4465],
      "std": [0.247, 0.2435, 0.2616]
    }
  },
  "dataset": {
    "id": "cifar10_test",
    "name": "cifar10_test",
    "task": "classification",
    "domain": "computer_vision",
    "num_classes": 10,
    "input_dimensionality": [3, 32, 32],
    "repository": "benchmark_assets/datasets/cifar10_test",
    "num_samples": 5,
    "batch_size": 32,
    "num_workers": 0
  },
  "attacks": [
    {
      "id": "deepfool",
      "name": "deepfool",
      "parameters": [
        {
          "id": "max_iters",
          "name": "max_iters",
          "default": 3
        }
      ],
      "task": "classification"
    },
    {
    "id": "fuap",
    "name": "fuap",
    "parameters": [
        {
        "id": "max_iters",
        "name": "max_iters",
        "default": 3
        }
    ],
    "task": "classification"
    }
  ],
  "metrics": [
    {
      "id": "accuracy",
      "name": "accuracy",
      "parameters": [],
      "task": "classification"
    },
    {
      "id": "misclassification",
      "name": "misclassification",
      "parameters": [],
      "task": "classification"
    },
    {
      "id": "robustness",
      "name": "robustness",
      "parameters": [],
      "task": "classification"
    }
  ],
  "options": {
    "overwrite": true,
    "num_images_to_save": 10,
    "save_perturbation": true,
    "gpu": true,
    "output_path": "benchmark_out",
    "use_ray": false,
    "num_workers": 1,
    "num_gpus_per_worker": 1.0,
    "create_pdf": false,
    "targeted": false
  }
}
```
