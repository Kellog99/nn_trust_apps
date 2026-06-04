"""Privacy protocol resolution and attack-construction helpers.

Includes protocol resolution from registered attack metadata and helpers for
constructing attack arguments. Supports nn_trust's privacy benchmarking protocol
registry and attack construction pipeline.
"""

import torch
from nn_trust.attack import AttackFactory

from .job_models import PrivacyJobConfig, PrivacyProtocolId


def resolve_requested_or_registered_protocol(job: PrivacyJobConfig) -> PrivacyProtocolId:
    """Resolve the privacy protocol from registered attack metadata."""
    info = AttackFactory.get_info(job.attack.attack_id)
    protocol = PrivacyProtocolId(str(getattr(info.privacy_type, "value", info.privacy_type)))
    if job.protocol is not None and job.protocol != protocol:
        raise ValueError(
            f"Configured protocol '{job.protocol}' does not match "
            f"registered protocol '{protocol}' for attack '{job.attack.attack_id}'."
        )
    return job.protocol or protocol


def build_privacy_attack_kwargs(
    job: PrivacyJobConfig,
    *,
    model: torch.nn.Module,
    device: torch.device,
    task,
    verbose: bool = False,
    extra_attack_kwargs: dict | None = None,
) -> dict:
    """Build kwargs for AttackFactory.create(...)."""
    params = dict(job.attack.attack_params)
    if extra_attack_kwargs:
        params.update(extra_attack_kwargs)
    return {
        "class_id": job.attack.attack_id,
        "model": model,
        "device": device,
        "task": task,
        "verbose": verbose,
        **params,
    }

