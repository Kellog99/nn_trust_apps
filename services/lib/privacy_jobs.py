from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from uuid import uuid4

from models import ServerConfig
from models.privacy import PrivacyAttackProps
from benchmarking.privacy.job_models import PrivacyAttackPayload, PrivacyDatasetConfig, PrivacyJobConfig, \
    PrivacyTargetModelConfig, PrivacyTargetTrainingOverrideConfig, RuntimeOptionConfig
from benchmarking.privacy.loading import resolve_privacy_protocol
from benchmarking.privacy.membership_inference import run_membership_inference_job
from benchmarking.privacy.model_inversion import run_model_inversion_job
from benchmarking.privacy.metrics import PrivacyExecutionResult
from benchmarking.privacy.persistence import (
    load_privacy_execution_result,
    privacy_result_exists,
    save_privacy_execution_result,
)
from benchmarking.privacy.property_inference import run_property_inference_job
from benchmarking.privacy.reconstruction import run_reconstruction_job

_executor = ThreadPoolExecutor(max_workers=1)
_jobs_lock = Lock()

_RUNNERS = {
    "membership_inference": run_membership_inference_job,
    "model_inversion": run_model_inversion_job,
    "property_inference": run_property_inference_job,
    "reconstruction": run_reconstruction_job,
}


@dataclass
class PrivacyJobRecord:
    config: PrivacyJobConfig
    future: Future[PrivacyExecutionResult]


_jobs: dict[str, PrivacyJobRecord] = {}


def execute_privacy_job(job: PrivacyJobConfig) -> PrivacyExecutionResult:
    protocol = resolve_privacy_protocol(job)
    if job.options.load_results and privacy_result_exists(job, protocol):
        return load_privacy_execution_result(job, protocol)
    if privacy_result_exists(job, protocol) and not job.options.overwrite:
        raise FileExistsError("Result exists. Set overwrite=true or load_results=true.")
    runner = _RUNNERS.get(protocol.value)
    if runner is None:
        raise NotImplementedError(f"Privacy protocol '{protocol.value}' is not implemented.")
    result = runner(job)
    save_privacy_execution_result(job, protocol, result)
    return result


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
    with _jobs_lock:
        _jobs[job_id] = PrivacyJobRecord(
            config=job_config,
            future=_executor.submit(execute_privacy_job, job_config),
        )
    return job_id


def get(job_id: str) -> PrivacyJobRecord | None:
    with _jobs_lock:
        return _jobs.get(job_id)
