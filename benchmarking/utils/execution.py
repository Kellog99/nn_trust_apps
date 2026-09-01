from logging import Logger
from pathlib import Path
from typing import Any, Iterator, Optional

import ray
import torch
from torch.utils.data import DataLoader
from tqdm.asyncio import tqdm

from benchmarking.utils import evaluate_attack
from models import JobResult
from nn_trust import ModelAdapter, StatisticComposer


def _run_single_attack(
        attack: dict[str, Any],
        model: ModelAdapter,
        dataloader: DataLoader,
        statistics: StatisticComposer,
        device: torch.device,
        output_path: Path,
        log: Optional[Logger],
        max_saved_elements: int = 10
) -> JobResult:
    """
    Runs evaluate_attack for a single attack config, always returning a JobResult.
    Never raises: failures are captured in JobResult.error.
    """
    atk_id: str = attack.get("id", None) or attack.get("name", "unknown")
    try:
        return evaluate_attack(
            dataloader=dataloader,
            model=model,
            attack=attack,
            statistics=statistics,
            device=device,
            verbose=False,
            output_path=output_path,
            max_saved_elements=max_saved_elements if max_saved_elements is not None else 10,
        )
    except Exception as exc:  # noqa: BLE001 - intentional: isolate per-attack failures
        if log is not None:
            log.exception(f"Attack '{atk_id}' failed")
        return JobResult(
            id=atk_id,
            result=None,
            parameters=None,
            error=str(exc),
        )


def _iter_local(
        model: ModelAdapter,
        dataloader: DataLoader,
        attacks: list[dict[str, Any]],
        statistics: StatisticComposer,
        device: torch.device,
        output_path: Path,
        max_saved_elements: int = 10,
        log: Optional[Logger] = None,
) -> Iterator[JobResult]:
    """
    Serial execution: yields one JobResult per attack, in order.
    """
    if not attacks:
        if log is not None:
            log.warning("_iter_local: 'attacks' is empty, no jobs will run")
        return
    pbar = tqdm(attacks)
    for attack in pbar:
        pbar.desc = f"{attack['id']}"
        yield _run_single_attack(
            attack=attack,
            model=model,
            dataloader=dataloader,
            statistics=statistics,
            device=device,
            output_path=output_path,
            max_saved_elements=max_saved_elements,
            log=log,
        )


def _iter_ray(
        model: ModelAdapter,
        dataloader: DataLoader,
        attacks: list[dict[str, Any]],
        statistics: StatisticComposer,
        device: torch.device,
        output_path: Path,
        max_saved_elements: Optional[int | dict[str, int]] = None,
        log: Optional[Logger] = None,
) -> Iterator[JobResult]:
    """
    Distributed execution via Ray: launches one remote task per attack,
    yields JobResults as they complete.
    """

    if not attacks:
        if log is not None:
            log.warning("_iter_ray: 'attacks' is empty, no jobs will run")
        return

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    remote_run = ray.remote(_run_single_attack)

    futures = [
        remote_run.remote(
            attack=attack,
            model=model,
            dataloader=dataloader,
            statistics=statistics,
            device=device,
            output_path=output_path,
            max_saved_elements=max_saved_elements,
            log=None,  # Logger objects are typically not picklable/serializable across Ray workers
        )
        for attack in attacks
    ]

    while futures:
        done, futures = ray.wait(futures, num_returns=1)
        yield ray.get(done[0])
