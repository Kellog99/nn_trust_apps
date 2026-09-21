import json
import inspect
from datetime import datetime
from logging import Logger
from pathlib import Path
from typing import List, Optional, Any

import torch
from torch.utils.data import DataLoader

from benchmarking.executor import BenchmarkExecutor
from models import BenchmarkOptionConfig, ModelInfo, DatasetInfo, ModelReportProps, RegisteredObject
from models.reports import ReportMetricsProps, ReportAttackProps
from nn_trust import AttackFactory as AF, StatisticComposer, StatisticsFactory as SF, ModelAdapter, Task
from utils import load_model, get_dataloader
from utils.load_dataset import get_transformation


def create_benchmark_id() -> str:
    """Create the identifier shared by a benchmark's tasks and output files."""
    return datetime.now().strftime("%Y%m%dT%H%M%S_%f")


_REGISTERED_OBJECT_METADATA = {
    "name",
    "description",
    "task",
    "knowledge",
    "objective",
    "privacy_type",
}


def _normalize_specs(
        items: Optional[list[RegisteredObject] | list[dict]],
) -> dict[str, dict[str, Any]]:
    """Convert API metadata or compact execution dictionaries to factory specs."""
    specs: dict[str, dict[str, Any]] = {}

    for item in items or []:
        if isinstance(item, RegisteredObject):
            specs[item.id] = {
                parameter.id: parameter.default
                for parameter in item.parameters
            }
            continue

        data = dict(item)
        object_id = str(data.pop("id"))
        parameters = data.pop("parameters", None)
        if parameters is not None:
            if isinstance(parameters, dict):
                specs[object_id] = dict(parameters)
            else:
                specs[object_id] = {
                    str(parameter["id"]): parameter.get("default")
                    for parameter in parameters
                }
        else:
            specs[object_id] = {
                key: value
                for key, value in data.items()
                if key not in _REGISTERED_OBJECT_METADATA
            }

    return specs


def run_benchmark(
        options: BenchmarkOptionConfig,
        models: List[ModelInfo],
        datasets: List[DatasetInfo],
        attacks: Optional[list[RegisteredObject] | list[dict]] = None,
        metrics: Optional[list[RegisteredObject] | list[dict]] = None,
        log: Optional[Logger] = None,
        benchmark_id: Optional[str] = None,
) -> list[ModelReportProps]:
    """
    This function takes as input a full benchmark configuration and executes the benchmark.
    If no attacks are given, only the identity baseline is run.
    """
    #################################### 1. Validate Items ####################################
    ##### 1.1 Datasets & models existence
    for dataset in datasets:
        if dataset.repository is None:
            raise ValueError("The path to the dataset repository is required.")
        if not Path(dataset.repository).expanduser().exists():
            raise FileNotFoundError(f"Dataset source path {dataset.repository} does not exist.")
    for model in models:
        if model.repository is None:
            raise ValueError("The path to the model repository is required.")
        if not Path(model.repository).expanduser().exists():
            raise FileNotFoundError(f"Model path {model.repository} does not exist.")

    ##### 1.2 Attacks: normalize and always include the identity baseline
    available_attacks = set(AF.get_list_classes())
    attack_specs = {
        attack_id: parameters
        for attack_id, parameters in _normalize_specs(attacks).items()
        if attack_id in available_attacks
    }
    attack_specs.setdefault("identitybaseline", {})

    ##### 1.3 Metrics: normalize to RegisteredObject
    metric_specs = _normalize_specs(metrics)
    if not metric_specs:
        raise ValueError("At least one benchmark metric must be selected.")

    #################################### 2. Prepare Execution ####################################
    benchmark_id = benchmark_id or create_benchmark_id()
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() and options.gpu else "cpu")
    base_output_path: str = options.output_path + f"/{benchmark_id}"

    list_reports: list[ModelReportProps] = []
    for model_cnf in models:
        task: Task = model_cnf.task if isinstance(model_cnf.task, Task) else Task.from_str(model_cnf.task)
        model: ModelAdapter = load_model(
            model_id=model_cnf.id or model_cnf.name,
            model_type=model_cnf.model_type,
            model_path=model_cnf.repository,
            api_url=model_cnf.api,
            task=task,
            device=device,
        )
        transform = get_transformation(transformation=model_cnf.transformation)

        for dataset_cnf in datasets:
            if dataset_cnf.repository is None:
                raise ValueError("No dataset to load.")

            dataloader: DataLoader = get_dataloader(
                dataset_type=dataset_cnf.dataset_type,
                dataset_path=dataset_cnf.repository,
                dataset_info=dataset_cnf,
                batch=dataset_cnf.batch_size,
                subset=options.subset,
                transform=transform,
                num_workers=dataset_cnf.num_workers,
                name=dataset_cnf.name,
                folder_data=dataset_cnf.folder_data,
                **(
                    {
                        "image_column": dataset_cnf.parquet_info.image_column,
                        "image_key": dataset_cnf.parquet_info.image_key,
                        "label_column": dataset_cnf.parquet_info.label_column,
                    }
                    if dataset_cnf.parquet_info is not None
                    else {}
                ),
            )

            #################### Defining the Statistic Composer ####################
            num_classes = model_cnf.num_classes
            if num_classes is None:
                batch, _ = next(iter(dataloader))
                with torch.no_grad():
                    num_classes = model(batch.to(device)).shape[-1]

            available_metrics = set(SF.get_list_classes(task={task}))
            selected_metrics: dict[str, dict] = {}
            for metric_id, parameters in metric_specs.items():
                if metric_id not in available_metrics:
                    continue

                metric_parameters = {
                    **parameters,
                    "num_classes": num_classes,
                }
                metric_class = SF.get_info(metric_id).class_type
                if "model" in inspect.signature(metric_class).parameters:
                    metric_parameters["model"] = model
                selected_metrics[metric_id] = metric_parameters

            statistics_composer = StatisticComposer(
                statistics=selected_metrics,
                device=device,
            )

            ######### 3.1 Start execution #########
            # The report path is benchmark_id / model_id / dataset_id
            report_path: Path = Path(base_output_path).expanduser().resolve() / model_cnf.id / dataset_cnf.id
            executor = BenchmarkExecutor(
                verbose=options.verbose,
                benchmark_id=benchmark_id,
                output_path=report_path,
            )
            results: dict[str, ReportAttackProps] = executor.execute_jobs(
                model=model,
                dataloader=dataloader,
                attacks=attack_specs,
                statistics=statistics_composer,
                device=device,
                log=log,
                max_saved_elements=options.max_saved_elements or 1,
            )
            global_metrics: dict = statistics_composer.compute_aggregator()

            # Global metrics: the identity baseline carries the requested performance
            # metrics; aggregator values (if any) override their identity-baseline
            # counterparts.
            identity: ReportAttackProps = results.pop("identitybaseline")
            report_metrics: dict = identity.metrics.model_dump(exclude_none=True)
            report_metrics["num_samples"] = len(dataloader.dataset)
            report_metrics.update(global_metrics)

            model_report = ModelReportProps(
                info=model_cnf,
                metrics=ReportMetricsProps.model_validate(report_metrics),
                attacks=results,
            )
            list_reports.append(model_report)

            print(report_path / "report.json")
            ######### saving the results #########
            with open(report_path / "report.json", "w") as f:
                json.dump(model_report.model_dump(), f)
            print("report saved")
            if log:
                log.info(
                    "Prepared job(s): %d model(s) x %d dataset(s) x %d attack(s).",
                    len(models), len(datasets), len(attack_specs),
                )

    return list_reports
