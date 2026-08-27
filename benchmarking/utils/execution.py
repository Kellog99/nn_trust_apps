from pathlib import Path
from typing import Iterator, Optional, Any

import ray
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from benchmarking.utils import evaluate_attack
from models import JobResult
from nn_trust import StatisticComposer, ModelAdapter


def _iter_local(
        model: ModelAdapter,
        dataloader: DataLoader,
        attacks: list[dict[str, Any]],
        statistics: StatisticComposer,
        device: torch.device = torch.device("cpu"),
        output_path: Optional[str | Path] = None,
        verbose: bool = True,
        **kwargs
) -> Iterator[JobResult]:
    """
    Execute all the attacks locally in a serialized way.
    """
    pbar = tqdm(attacks, disable=not verbose, desc="Running jobs")
    for atk in pbar:
        atk_id = atk.get('id', 'None')
        pbar.desc = f"Running {atk_id}"
        try:
            yield evaluate_attack(
                dataloader=dataloader,
                model=model,
                attack=atk,
                statistics=statistics,
                device=device,
                output_path=output_path,
            )
        except Exception as e:
            yield JobResult(
                id=atk_id,
                error=e
            )


def _iter_ray(
        model: ModelAdapter,
        dataloader: DataLoader,
        attacks: list[dict[str, Any]],
        statistics: StatisticComposer,
        device: torch.device = torch.device("cpu"),
        output_path: Optional[str | Path] = None,
        verbose: bool = True,
        num_gpus_per_job: float = 0.6,
        **kwargs
) -> Iterator[JobResult]:
    remote_execute_job = ray.remote(num_gpus=num_gpus_per_job)(evaluate_attack)
    pending: dict[ray.ObjectRef, dict[str, Any]] = {
        remote_execute_job.remote(
            dataloader=dataloader,
            model=model,
            attack=atk,
            statistics=statistics,
            output_path=output_path,
            device=device,
        ): atk
        for atk in attacks
    }

    pbar = tqdm(
        total=len(pending),
        disable=not verbose,
        desc="Running jobs"
    )
    with pbar:
        while pending:
            done, _ = ray.wait(list(pending.keys()), num_returns=1)
            ref = done[0]
            atk = pending.pop(ref)
            pbar.update(1)
            try:
                yield ray.get(ref)
            except Exception as e:
                yield JobResult(
                    id=atk.get('id', 'None'),
                    error=e
                )
