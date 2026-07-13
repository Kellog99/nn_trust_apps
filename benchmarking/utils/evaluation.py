import json
import logging
import os
import pathlib
import pickle
import traceback
from pathlib import Path

import torch
from tqdm.auto import tqdm

from nn_trust.attack import AttackFactory as EAF
from nn_trust.core import Task, ModelAdapter
from nn_trust.evaluation.composer import ConfigStatisticComposer, StatisticComposer
from nn_trust.loss.loss_composer import LossComposer
from nn_trust.target import AvoidOnehotTarget

def evaluate_attack(
        dataloader: torch.utils.data.DataLoader,
        model: ModelAdapter,
        attack_config: dict,
        statistics: list[dict],
        device: torch.device,
        num_classes: int,
        verbose: bool = False,
        tracker=None,
        benchmark_id: str = None,
) -> dict:
    """
    Evaluate the model on the attack that is passed.

        Args:
            atk: the attack that has to be performed.
    """
    try:
        # INIT MODEL , DATA, STATISTIC_COMPOSER, ATTACK
        ## 1. STATISTIC_COMPOSER
        statistics_composer = StatisticComposer(config=ConfigStatisticComposer(
            statistics=statistics,
            num_classes=num_classes
        ))
        ## 2. ATTACK
        atk_name = attack_config.pop("name")
        atk_id = attack_config.pop("id", atk_name)
        if attack_config.get("losses"):
            # If losses are specified, convert them to Loss objects
            attack_config['loss'] = LossComposer(
                loss=attack_config['losses'],
                loss_weights=attack_config.get('loss_weights', [1.0] * len(attack_config['losses'])),
            )

        targeted = attack_config.pop("targeted", False)

        atk = EAF.create(
            atk_name,
            model=model,
            device=device,
            task=Task.Classification,
            targeted=targeted,
            **attack_config
        )
        atk.name = atk_id
        if model.task not in atk.task:
            raise ValueError(
                f"\U0001F928 Attack {atk_name} does not support Model {model.name} task {model.task}.")

        ### PREPARE EXECUTION
        if verbose:
            progress_bar = enumerate(tqdm(dataloader, desc=f"Attack {atk.name} for model {model.name}"))
        else:
            progress_bar = enumerate(dataloader)

        # tracker.create_task.remote(f"{atk_id}_{benchmark_id}","attack", benchmark_id = benchmark_id, num_tasks=num_tasks)
        for idx, (batch, label, element_info) in progress_bar:
            if tracker:
                tracker.update_progress.remote(f"{atk_id}_{benchmark_id}",
                                               status="in_progress",
                                               progress=int((idx / len(dataloader)) * 100),
                                               message=f"Processing batch {idx + 1}/{len(dataloader)}")
            batch = batch.to(device)
            label = label.to(device)
            if targeted:
               target_classes = (label + 1) % num_classes
               target = torch.nn.functional.one_hot(target_classes,num_classes=num_classes).float().to(batch.device)
            else:
               target = AvoidOnehotTarget(num_classes=num_classes)(label.tolist()).to(batch.device)

            x_adv = atk.generate(
                x=batch,
                y=target
            ).detach()

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

                    if targeted:
                      target_classes = target_classes[correct_mask]
            else:
                continue  # skip iteration, no statistic update for this batch

            input_stat = {
                'x_adv': x_adv.detach(),
                'x': batch.detach(),
                'y': label,
                "y_target": target_classes if targeted else label,
                'out': out,
                'out_adv': out_adv,
                'y_pred': y_pred,
                'y_pred_adv': y_pred_adv
            }

            if targeted:
              input_stat["y_target"] = target_classes

            statistics_composer.update(**input_stat)

        if tracker:
            tracker.update_progress.remote(f"{atk_id}_{benchmark_id}", status="completed", progress=100,
                                           message=f"Completed attack {atk_id}")
            
        saved_statistics = []
        for metric in statistics:
            metric = dict(metric)
            metric.pop("model", None)
            saved_statistics.append(metric)

        return {
            "statistics": statistics_composer.compute(),
            "statistics_states": statistics_composer.get_raw_state(),
            "info": {
                'name': model.name,
                'parameters': sum([param.numel() for param in model.parameters()]),
                'classes': num_classes,
                'dimensionality': batch.shape,
                'statistics': saved_statistics,
                # if model metadata do not exist, save an empty dictionary
                "model_info": {**getattr(model, "metadata", {}), "num_classes": num_classes,"parameters": sum(param.numel() for param in model.parameters())},
                # if dataloader metadata do not exist, save an empty dictionary
                "dataset_info": getattr(dataloader, "metadata", {})
            }
        }
    except Exception as e:
        logging.error(f"Error during evaluation of attack {attack_config.get('name', 'unknown')} : {e}")
        traceback.print_exc()
        if tracker:
            tracker.update_progress.remote(
                f"{atk_id}_{benchmark_id}",
                status="completed",
                progress=50,
                message=f"Failed attack {atk_id} with error {e}"
            )
        raise e


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
