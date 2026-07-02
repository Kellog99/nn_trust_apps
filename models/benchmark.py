from pydantic import BaseModel

from models.info import ModelInfo, DatasetInfo
from models.model import RegisteredObject


class BenchmarkOptionConfig(BaseModel):
    load_results: bool
    overwrite: bool
    num_images_to_save: int
    save_perturbation: bool
    gpu: bool
    output_path: str
    use_ray: bool = False
    num_workers: int = 1
    num_gpus_per_worker: float = 1.0
    create_pdf: bool = False


# This class is for handling the type of the benchmark's service input
class BenchmarkExecutionConfig(BaseModel):
    model: ModelInfo
    dataset: DatasetInfo
    attacks: list[RegisteredObject]
    metrics: list[RegisteredObject]
    options: BenchmarkOptionConfig
