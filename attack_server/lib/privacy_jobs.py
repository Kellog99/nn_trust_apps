from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from uuid import uuid4

from attack_server.models.main_model import ServerConfig
from attack_server.models.privacy import PrivacyAttackProps
from benchmarking.privacy.execution import execute_privacy_job
from benchmarking.privacy.job_models import (
    PrivacyAttackPayload,
    PrivacyDatasetConfig,
    PrivacyJobConfig,
    PrivacyTargetModelConfig,
    PrivacyTargetTrainingOverrideConfig,
)
from benchmarking.privacy.metrics import PrivacyExecutionResult
from benchmarking.privacy.job_models import RuntimeOptionConfig

_executor = ThreadPoolExecutor(max_workers=1)
_jobs_lock = Lock()


@dataclass
class PrivacyJobRecord:
    config: PrivacyJobConfig
    future: Future[PrivacyExecutionResult]


_jobs: dict[str, PrivacyJobRecord] = {}


def to_privacy_job_config(
    body: PrivacyAttackProps,
    config: ServerConfig,
    *,
    device: str = "cpu",
) -> PrivacyJobConfig:
    model = body.model
    return PrivacyJobConfig(
        dataset=PrivacyDatasetConfig(
            dataset_id=body.dataset.id,
            root=Path(body.dataset.root or "./data").expanduser(),
            task_attr=body.dataset.task_attr,
            use_embeddings=body.dataset.use_embeddings,
            max_samples=body.dataset.max_samples,
            seed=body.dataset.seed,
        ),
        target_model=PrivacyTargetModelConfig(
            model_id=model.id,
            source_type=model.source_type,
            training_recipe_id=model.training_recipe_id,
            checkpoint_path=Path(model.checkpoint_path).expanduser() if model.checkpoint_path else None,
            shadow_model_id=model.shadow_model_id,
            property_ratio=model.property_ratio,
            property_name=model.property_name,
            property_target_ratio=model.property_target_ratio,
            training_overrides=(
                PrivacyTargetTrainingOverrideConfig.model_validate(model.training_overrides)
                if model.training_overrides else None
            ),
        ),
        attack=PrivacyAttackPayload(
            attack_id=body.attack.id,
            attack_params={p.id: p.default for p in body.attack.parameters},
        ),
        options=RuntimeOptionConfig(
            overwrite=True,
            output_path=str(Path(config.path_tmp_files).expanduser() / "privacy"),
            output_format="json",
            gpu=device == "cuda",
            mode="local_serial",
        ),
        verbose=True,
    )


def submit(job_config: PrivacyJobConfig) -> str:
    job_id = uuid4().hex
    job_record = PrivacyJobRecord(
        config=job_config,
        future=_executor.submit(execute_privacy_job, job_config),
    )
    with _jobs_lock:
        _jobs[job_id] = job_record
    return job_id


def get(job_id: str) -> PrivacyJobRecord | None:
    with _jobs_lock:
        return _jobs.get(job_id)
