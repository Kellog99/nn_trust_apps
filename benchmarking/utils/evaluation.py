import json
import os
import pathlib
import pickle
from pathlib import Path

import torch
from tqdm.auto import tqdm

from models.benchmark import AttackEvaluation
from nn_trust import ModelAdapter, StatisticComposer
from nn_trust.attack import EvasionAttack
from nn_trust.target import AvoidOnehotTarget


def evaluate_attack(
        dataloader: torch.utils.data.DataLoader,
        model: ModelAdapter,
        attack: EvasionAttack,
        statistics: StatisticComposer,
        device: torch.device,
        num_classes: int,
        verbose: bool = False,
) -> AttackEvaluation:
    """
    Evaluate the model's vulnerability on the attack that is passed.
    Since this is for performance purpose, it is assumed that it is not targeted

        Args:
            dataloader: dataset to use
            model: target model
            attack: Attack to do on the (model, dataset)
            statistics: The statistic composer that computes all the metrics that are required
            num_classes
            verbose
            benchmark_id
            device: device where the computation will be done
    """

    ### PREPARE EXECUTION
    atk_id: str = attack.__class__.__name__.lower().removesuffix("attack")

    if verbose:
        progress_bar = enumerate(tqdm(dataloader, desc=f"Attack {repr(attack)} for model {model.name}"))
    else:
        progress_bar = enumerate(dataloader)

    # tracker.create_task.remote(f"{atk_id}_{benchmark_id}","attack", benchmark_id = benchmark_id, num_tasks=num_tasks)
    for idx, (batch, label) in progress_bar:

        batch = batch.to(device)
        label = label.to(device)

        target = AvoidOnehotTarget(num_classes=num_classes)(label.tolist()).to(batch.device)

        ############## Generating the adversarial image ##############
        x_adv = attack.generate(
            x=batch,
            y=target
        ).detach()
        ##############################################################

        with torch.no_grad():
            out = model(batch)
            out_adv = model(x_adv)

        y_pred_adv = out_adv.argmax(dim=-1)
        y_pred = out.argmax(dim=-1)

        # adapt metrics counting for reference or standard attack
        is_reference = atk_id == "reference"
        correct_mask = torch.eq(label, y_pred)
        if is_reference:
            y_pred = label

        elif torch.any(correct_mask):
            if not torch.all(correct_mask):
                label = label[correct_mask]
                x_adv = x_adv[correct_mask]
                batch = batch[correct_mask]
                out = out[correct_mask]
                out_adv = out_adv[correct_mask]
                y_pred = y_pred[correct_mask]
                y_pred_adv = y_pred_adv[correct_mask]

        else:
            continue  # skip iteration, no statistic update for this batch

        input_stat = {
            'x_adv': x_adv.detach(),
            'x': batch.detach(),
            'y': label,
            'y_target': label,
            'y_pred': y_pred,
            'y_pred_adv': y_pred_adv
        }

        statistics.update(**input_stat)

    out = AttackEvaluation(
        statistics=statistics.compute(),
        statistics_states=statistics.get_raw_state()
    )
    return out


def read_results_from_disk(results_dir: str | pathlib.Path):
    r"""
    Read from disk back to Evaluator results object structure
    The directory from which results are read are the same kind of the target of `save_`

    >>>results = dict(
    ...     info,
    ...     results=dict(attacks=dict(statistics, statistics_states)),
    ...     aggregate_statistics=optional
    ...)
    """
    results_dir = Path(results_dir)
    results = {"attacks": {}}
    attacks_dir = [attack_dir for attack_dir in results_dir.iterdir() if attack_dir.is_dir()]
    for attack_dir in attacks_dir:
        with open(attack_dir / "statistics.json", "r") as fmetric:
            statistics = json.load(fmetric)
        with open(attack_dir / "statistics_states.pkl", "rb") as fdata:
            statistics_states = pickle.load(fdata)
        results["attacks"][attack_dir.name] = {
            "statistics": statistics,
            "statistics_states": statistics_states
        }
        with open(attack_dir / "info.json", "r") as f:
            results["info"] = json.load(f)
    if "aggregate_statistics.json" in results_dir.iterdir():
        with open(results_dir / "aggregate_statistics.json", "r") as f:
            results["aggregate_statistics"] = json.load(f)
    return results


def aggregate_attacks_statistics(
        statistics_composer: StatisticComposer,
        results: dict
) -> dict:
    """
    Use statistic composer in aggregation mode to aggregate statistics states and
    compute aggregated results
    """
    for attack, attack_results in results['attacks'].items():
        if not attack == "reference":
            statistics_composer.update_aggregate(attack_results["statistics_states"])
    aggregate_metrics = statistics_composer.compute()
    statistics_composer.reset()
    return aggregate_metrics


def make_json_serializable(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.numel() == 1:
            return value.item()
        return value.tolist()

    if isinstance(value, torch.Size):
        return list(value)

    if isinstance(value, torch.device):
        return str(value)

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {
            str(key): make_json_serializable(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [make_json_serializable(item) for item in value]

    if isinstance(value, tuple):
        return [make_json_serializable(item) for item in value]

    # if json can already save it, keep it. If not, convert it to a string
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def save_attack_result(
        benchmark_id: str,
        atk_result: dict,
        atk_id: str,
        dataset_name: str,
        model_name: str,
        root_path: str | pathlib.Path) -> None:
    """
    Save single attack results from static method -evaluate_attack- to a JSON file
    """
    output_path = Path(benchmark_id) / f"{model_name}_{dataset_name}" / atk_id
    if root_path:
        output_path = Path(root_path) / output_path
    os.makedirs(output_path, exist_ok=True)
    with open(output_path / "statistics.json", 'w') as f:
        json.dump(make_json_serializable(atk_result["statistics"]), f)
    with open(output_path / "statistics_states.pkl", 'wb') as f:
        pickle.dump(atk_result["statistics_states"], f)
    with open(output_path / "info.json", 'w') as f:
        json.dump(make_json_serializable(atk_result["info"]), f)
