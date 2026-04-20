from typing import Annotated, Literal, List, Dict

from pydantic import BaseModel, Field

class BenchmarkEvaluationConfig(BaseModel):
    statistics: list[dict] | None = Field(default_factory=lambda x: [])

class BenchmarkOptionConfig(BaseModel):
    load_results: bool
    overwrite: bool
    num_images_to_save: int
    save_perturbation: bool
    gpu: bool
    output_path: str
    output_format: str
    mode: Literal["local_serial", "local_ray"]
    num_workers: int = 1
    num_gpus_per_worker: float = 1.0
    executor_type: Literal["none", "ray"] = "none"


class BenchmarkDatasetTransformConfig(BaseModel):
    size: int | None = None
    crop: int | None = None
    transform_id: str
    mean: List[float]
    std: List[float]


class BenchmarkDatasetConfig(BaseModel):
    name: str
    num_classes: int
    subset: int
    batch: int
    type_dataset: int
    num_workers: int
    source_path: str
    transform_config: BenchmarkDatasetTransformConfig


class BenchmarkModelsConfig(BaseModel):
    name: str | None = None
    model_path: str | None = None
    type: str | None = None
    pretrained: bool | None = None
    num_classes: int | None = None
    task: str | None = None
    weights_path: str | None = None
    input_size: int | None = None


class BenchmarkAttackConfig(BaseModel):
    name: str
    id: str | None = None
    max_iters: int | None = None
    losses: List[str] | None = None
    optim_lr: float | None = None
    optim_momentum: float | None = None
    optim_nesterov: bool | None = None
    verbose: bool | None = False


class BenchmarkConfig(BaseModel):
    evaluation: BenchmarkEvaluationConfig
    options: BenchmarkOptionConfig
    datasets: List[BenchmarkDatasetConfig]
    models: List[BenchmarkModelsConfig]
    attacks: List[BenchmarkAttackConfig]