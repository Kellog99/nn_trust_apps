from typing import Any, Optional, TypedDict

from pydantic import BaseModel, ConfigDict, Field

from models.info import ModelInfo, DatasetInfo
from models.model import RegisteredObject
from models.reports import ParameterLog


class TaskStatus(TypedDict):
    """Dictionary representation of an attack's benchmark task."""

    benchmark_id: str
    attack_id: str
    name: str
    status: str
    progress: int
    error: str | None


class BenchmarkOptionConfig(BaseModel):
    """
    This class contains all the global variables for the benchmark
    """
    overwrite: bool = True
    save_perturbation: bool = True
    variables_to_save: list[str] = Field(default_factory=lambda: ["original_input", "res"])
    max_saved_elements: Optional[int] = None
    verbose: bool = True
    subset: Optional[int] = None
    gpu: bool = True
    output_path: str = "~/Desktop/StableAI/benchmark_repository"
    use_ray: bool = False
    num_workers: int = 1
    num_gpus_per_worker: float = 1.0
    create_pdf: bool = False
    targeted: bool = False


# This class is for handling the type of the benchmark's service input
class BenchmarkExecutionConfig(BaseModel):
    attacks: list[RegisteredObject]
    metrics: list[RegisteredObject]
    # The web client sends the selected model and dataset IDs.  The router
    # resolves those IDs to their repository metadata before starting a job.
    model: ModelInfo
    dataset: DatasetInfo
    options: BenchmarkOptionConfig = Field(default_factory=BenchmarkOptionConfig)


class JobResult(BaseModel):
    """
    Since this has to handle the errors to, only the id is required.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    id: str
    parameters: Optional[list[ParameterLog]] = None
    result: Optional[dict] = None
    error: Optional[BaseException | str] = None
