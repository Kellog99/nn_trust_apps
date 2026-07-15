from pydantic import BaseModel, ConfigDict

from models.info import ModelInfo, DatasetInfo
from models.model import RegisteredObject


class BenchmarkOptionConfig(BaseModel):
    overwrite: bool = True
    num_images_to_save: int = 10
    save_perturbation: bool = True
    gpu: bool = True
    output_path: str
    use_ray: bool = False
    num_workers: int = 1
    num_gpus_per_worker: float = 1.0
    create_pdf: bool = False
    targeted: bool = False


# This class is for handling the type of the benchmark's service input
class BenchmarkExecutionConfig(BaseModel):
    model: ModelInfo
    dataset: DatasetInfo
    attacks: list[RegisteredObject]
    metrics: list[RegisteredObject]
    options: BenchmarkOptionConfig

    # exemplary model_config 
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
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
                    "num_samples": 100,
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
                        },
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
                    "overwrite": True,
                    "num_images_to_save": 10,
                    "save_perturbation": True,
                    "gpu": True,
                    "output_path": "benchmark_out",
                    "use_ray": False,
                    "num_workers": 1,
                    "num_gpus_per_worker": 1.0,
                    "create_pdf": False,
                    "targeted": False
                }
            }
        }    
    )