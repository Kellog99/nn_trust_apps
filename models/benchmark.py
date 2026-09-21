import json
from pathlib import Path
from typing import Optional, TypedDict, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

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


class BenchmarkExecutionConfig(BaseModel):
    """
    This class is for handling the type of the benchmark's service input
    """
    attacks: list[RegisteredObject]
    metrics: list[RegisteredObject]
    model: ModelInfo
    dataset: DatasetInfo
    options: BenchmarkOptionConfig = Field(default_factory=BenchmarkOptionConfig)
    benchmark_id: Optional[str] = None


class JobResult(BaseModel):
    """
    Since this has to handle the errors too, only the id is required.
    """
    id: str
    parameters: Optional[list[ParameterLog]] = None
    result: Optional[dict] = None
    total: Optional[int] = None
    progress: Optional[int] = None
    iteration_time: Optional[float] = None
    execution_time: Optional[float] = None
    estimated_execution_time: Optional[float] = None
    status: Literal["pending", "in progress", "finished", "error"] = "pending"
    error: Optional[str] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def validate_status(self) -> "JobResult":
        if self.status == "finished":
            if self.result is None:
                raise ValueError("If the job is finished then there must be a result.")
            if self.total != self.progress:
                raise ValueError("The total number of elements must be the same as the one that have been seen.")
        if self.status == "error" and self.error is None:
            raise ValueError("If the job is in error state then `error` must be set.")
        return self

    def save(self, path: Path | str) -> "JobResult":
        """
        Persist this result as JSON at the given path and return self.
        """
        path = Path(path)

        if path.is_dir():
            raise IsADirectoryError(f"Expected a file path, got a directory: {path}")
        if not path.parent.exists():
            path.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() != ".json":
            raise ValueError(f"Expected a .json file, got: {path}")

        with open(path, "w") as f:
            json.dump(self.model_dump(), f)
        return self
