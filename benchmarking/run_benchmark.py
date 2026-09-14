import json
from datetime import datetime
from logging import Logger
from pathlib import Path
from typing import List, Optional, Any

import torch
from torch.utils.data import DataLoader

from benchmarking.executor import BenchmarkExecutor
from models import BenchmarkOptionConfig, ModelInfo, DatasetInfo, ModelReportProps, RegisteredObject
from models.reports import ReportMetricsProps, ReportAttackProps
from nn_trust import StatisticComposer, StatisticsFactory as SF, ModelAdapter, Task
from utils import load_model, get_dataloader
from utils.load_dataset import get_transformation


def create_benchmark_id() -> str:
    """Create the identifier shared by a benchmark's tasks and output files."""
    return datetime.now().strftime("%Y%m%dT%H%M%S_%f")


def _normalize_registered_objects(items: Optional[list]) -> List[RegisteredObject]:
    """Coerce a list of dict/RegisteredObject into RegisteredObject instances."""
    items = items or []
    return [
        item if isinstance(item, RegisteredObject) else RegisteredObject.model_validate(item)
        for item in items
    ]


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
    If no attacks are given, only the identity baseline is run; if no metrics are given, none are computed.
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

    ##### 1.2 Attacks: normalize to RegisteredObject, always include the identity baseline
    attack_list: List[RegisteredObject] = _normalize_registered_objects(attacks)
    if not any(attack.id == "identitybaseline" for attack in attack_list):
        attack_list.append(RegisteredObject(
            id="identitybaseline",
            name="identitybaseline",
            parameters=[],
            task=Task.Classification.name,
        ))
    # id -> {param_id: default} spec expected by the executor
    attack_specs: dict[str, dict[str, Any]] = {
        atk.id: {
            param.id: param.default
            for param in atk.parameters
        }
        for atk in attack_list
    }

    ##### 1.3 Metrics: normalize to RegisteredObject
    metric_list: List[RegisteredObject] = _normalize_registered_objects(metrics)

    #################################### 2. Prepare Execution ####################################
    benchmark_id = benchmark_id or create_benchmark_id()
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() and options.gpu else "cpu")
    base_output_path: str = options.output_path + f"/{benchmark_id}"

    executor = BenchmarkExecutor(
        verbose=options.verbose,
        benchmark_id=benchmark_id,
        use_ray=options.use_ray,
        output_path=base_output_path,
    )

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
            selected_metrics: dict[str, dict] = {
                metric.id: {
                    param.id: param.default
                    for param in metric.parameters
                }
                for metric in metric_list
                if metric in SF.get_list_classes(task={task})
            }
            statistics_composer = StatisticComposer(
                statistics=selected_metrics,
                device=device,
            )

            # 3.1 Start execution
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

            ######### saving the results #########
            model_dataset_path: Path = Path(
                base_output_path).expanduser().resolve() / f"{model_cnf.id}/{dataset_cnf.id}"
            model_dataset_path.mkdir(parents=True, exist_ok=True)
            with open(model_dataset_path / "report.json", "w") as f:
                json.dump(model_report.model_dump(), f)

            if log:
                log.info(
                    "Prepared job(s): %d model(s) x %d dataset(s) x %d attack(s).",
                    len(models), len(datasets), len(attack_list),
                )

    return list_reports
