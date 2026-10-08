import argparse
import math
from pathlib import Path
from typing import Annotated
from types import SimpleNamespace

import optuna
import torch
from annotated_types import Ge, Gt, Le, Lt
from torch.utils.data import DataLoader
from nn_trust.evaluation.metrics.detection import misdetection
from nn_trust.evaluation.metrics.basic import Misclassification
from nn_trust.evaluation.metrics.ssim import SSIM

import yaml
from benchmarking.utils.load_info import load_info
from benchmarking.utils.evaluation_pipeline import run_evaluation_pipeline
from models import ModelInfo, DatasetInfo
from nn_trust import AttackFactory, CVModelAdapter, EvasionAttack, Task
from nn_trust.attack.utils.detection import absolutize_items
from nn_trust.utils import Logger
from utils.dataset_utils import get_transform_dataset
from utils.load_dataset import get_dataloader
from utils.load_model import load_model


class Objective:
    """
    This class has the role to instantiate the attack and be used for finding the best parameters for the attack
    """

    def __init__(self,
                 model: CVModelAdapter,
                 task: Task,
                 attack: str,
                 dataloader: DataLoader,
                 max_number_parameters: int,
                 interval_length: float = 100.,
                 device=torch.device(
                     "cuda" if torch.cuda.is_available()
                     else "mps" if torch.mps.is_available()
                     else "cpu"
                 ),
                 p: Annotated[float, Ge(1.)] = 2.,
                 parameters: dict | None = None,
                 output_dir: str | Path = "hyperparameter_output",
                 **kwargs):
        self.model = model.to(device).eval()
        self.dataloader = dataloader
        self.task = task
        self.attack_name = attack
        self.device = device
        self.p = p
        self.parameters = dict(parameters or {})
        self.output_dir = Path(output_dir)
        if model.task != task:
            raise ValueError("The objective task must match model.task.")
        info = AttackFactory.get_info(attack)
        if task not in (Task.Classification, Task.Detection) or task not in info.task:
            raise ValueError(f"Attack {attack!r} does not support {task}.")
        if not issubclass(info.class_type, EvasionAttack):
            raise ValueError("Only evasion attacks support this tuning objective.")

        atk_params = AttackFactory.get_config_param(attack)
        self.params = {}
        for field_name, field_info in atk_params:
            if attack == "advyoloevasion" and field_name != "total_variation_weight":
                continue  # The inherited numerical-attack floats do not train the patch.
            if field_info.annotation == float:
                print(f"get the {field_name}")
                if len(self.params) >= max_number_parameters:
                    break

                metadata = field_info.metadata
                lower = next((m for m in metadata if isinstance(m, (Ge, Gt))), None)
                if lower is None:
                    raise ValueError(f"Missing lower bound for {field_name}")
                ge = lower.ge if isinstance(lower, Ge) else math.nextafter(lower.gt, math.inf)
                upper = next((m for m in metadata if isinstance(m, (Le, Lt))), None)
                if upper is None:
                    le = ge + interval_length
                else:
                    raw_upper = upper.le if isinstance(upper, Le) else upper.lt
                    if not math.isfinite(raw_upper):
                        le = ge + interval_length
                    else:
                        le = raw_upper if isinstance(upper, Le) else math.nextafter(raw_upper, -math.inf)
                self.params[field_name] = (ge, le)
                print(self.params)

    def __call__(self, trial) -> tuple[float, float]:
        params = {**self.parameters, **{
            name: trial.suggest_float(name, low, high)
            for name, (low, high) in self.params.items()
        }}
        attack = AttackFactory.create(
            class_id=self.attack_name, model=self.model, device=self.device, **params,
        )
        metric = None
        ssim = None

        targeted = (
            attack.config.targeted
            and self.attack_name not in {"advyoloevasion", "identitybaseline"}
        )
        
        def update_statistics(x, x_adv, y, y_pred, y_pred_adv, y_target, out, **kwargs):
            nonlocal metric, ssim
            if not torch.isfinite(x_adv).all():
                raise optuna.TrialPruned("Attack produced non-finite pixels.")

            if metric is None:
               match self.task:
                    case Task.Detection:
                        metric = misdetection(
                            device=self.device,
                            num_classes=out[1].shape[-1],
                            targeted=targeted,
                            label_target=attack.config.label_target,
                            w=x.shape[-1],
                            h=x.shape[-2],
                        ).to(self.device)
                    case Task.Classification:
                        metric = Misclassification(
                            device=self.device,
                            num_classes=out.shape[-1],
                            targeted=targeted,
                        ).to(self.device)
                    case _:
                        raise ValueError(f"Unsupported task {self.task}.")

            metric.update(
                x=x,
                x_adv=x_adv,
                y=y,
                y_pred=y_pred,
                y_pred_adv=y_pred_adv,
                y_target=y_target,
                **kwargs,
            )

            if ssim is None:
                ssim = SSIM(device=self.device).to(self.device)

            ssim.update(
                x=x, 
                x_adv=x_adv,
                **kwargs,
            )

        try:
            statistics = SimpleNamespace(
                update=update_statistics,
                get_raw_state=lambda: {},
                update_aggregate=lambda state: None,
            )
            run_evaluation_pipeline(
                model=self.model, attack=attack, attack_id=self.attack_name,
                dataloader=self.dataloader, statistics=statistics, device=self.device,
                logger=Logger(), output_path=self.output_dir / f"trial_{trial.number:05d}",
            )

            performance = metric.compute()
            if not math.isfinite(performance) or performance < 0:
                raise optuna.TrialPruned("Misclassification/Misdetection is undefined for this evaluation set.")
            ssim_value = ssim.compute()
            if not math.isfinite(ssim_value):
                raise optuna.TrialPruned("Non-finite SSIM.")
            return performance, ssim_value
        finally:
            attack.logger.close()



def get_args():
    """
    Argparser for the hyperparameters optimization
    """
    parser = argparse.ArgumentParser()

    # Paths
    parser.add_argument('--config_path', '-c',
                        type=str,
                        required=True,
                        help='Path to the benchmark YAML configuration')
    # Optuna parameters
    parser.add_argument('--n_trials', '-t',
                        type=int,
                        default=30,
                        help='Number of Optuna trials')

    # Checkpoints / Logging
    parser.add_argument('--output_dir', '-out',
                        type=str,
                        help='Directory for the Optuna SQLite database')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    ########################### Loading the configuration file ###########################
    with open(args.config_path) as f:
        config = yaml.safe_load(f)

    model_entries = config.get("models") or config.get("model")
    if not model_entries:
        raise ValueError("The YAML must contain a model or models entry.")
    model_entry = dict(model_entries[0])
    dataset_entry = dict(config["datasets"][0])
    #if args.model:
    #    model_entry["source_path"] = args.model
    #if args.source_path:
    #    dataset_entry["source_path"] = args.source_path
    model_info = load_info(model_entry, ModelInfo)
    dataset_info = load_info(dataset_entry, DatasetInfo)

    task = Task.from_str(model_info.task)
    if task not in (Task.Classification, Task.Detection):
        raise ValueError("This tuning objective supports classification and detection only.")

    device = torch.device("cuda" if torch.cuda.is_available() and config.get("options", {}).get("gpu", True) else "cpu")

    model = load_model(
        model_type=model_info.model_type,
        model_id=model_info.id,
        model_path=model_info.repository,
        api_url=model_info.api,
        task=task,
        device=device,
    )
    model.eval()

    dataloader = get_dataloader(
        dataset_path=dataset_info.repository,
        dataset_info=dataset_info,
        dataset_type=dataset_info.dataset_type,
        batch=dataset_info.batch_size,
        subset=config.get("options", {}).get("subset"),
        num_workers=dataset_info.num_workers,
        folder_data=dataset_info.folder_data,
        parquet_info=dataset_info.parquet_info,
        transform=get_transform_dataset(model_info.transformation),
        task=task,
        **({"new_shape": tuple(model_info.input_dimensionality[-2:])} if task == Task.Detection else {}),
    )
    attack_entries = config.get("attacks", [])
    attack_id = attack_entries[0]["id"]
    parameters = dict(next((entry for entry in attack_entries if entry["id"] == attack_id), {}))
    parameters.pop("id", None)
    #if args.max_iters is not None:
    #    parameters["max_iters"] = args.max_iters
    output_dir = Path(args.output_dir or ".")
    output_dir.mkdir(parents=True, exist_ok=True)
    objective = Objective(model=model,
                          attack=attack_id,
                          dataloader=dataloader,
                          task=task,
                          device=device,
                          parameters=parameters,
                          output_dir=output_dir,
                          max_number_parameters=100)

    ########################### OPTUNA ###########################
    study = optuna.create_study(
        storage=f"sqlite:///{output_dir.resolve() / f'db.{attack_id}.sqlite3'}",
        directions=["maximize", "maximize"],
        sampler=optuna.samplers.TPESampler(seed=1),
    )
    study.optimize(objective, n_trials=args.n_trials)

    min_attacked_performance = 0.80  # Example misclassification/misdetection ceiling

    candidates = [
        trial for trial in study.best_trials
        if trial.values[0] >= min_attacked_performance
    ]

    best = max(candidates, key=lambda trial: trial.values[1]) if candidates else None

    if best is None:
        print(f"No trial reached the minimum attack success of {str(min_attacked_performance)}.")
    else:
        best_parameters = {**parameters, **best.params}
        print("Selected trial:", best.number)
        print(f"Misdetection, SSIM: {best.values[0]:.6f}, {best.values[1]:.6f}")
        print("Attack parameters:", best_parameters)
    ##############################################################
