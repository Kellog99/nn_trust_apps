import json
from datetime import datetime
from logging import Logger
from pathlib import Path
from typing import List, Optional

import torch
from torch.utils.data import DataLoader

from benchmarking.executor import BenchmarkExecutor
from models import BenchmarkOptionConfig, ModelInfo, DatasetInfo, ModelReportProps
from models.reports import ReportMetricsProps, ReportAttackProps
from nn_trust import AttackFactory as AF, StatisticComposer, StatisticsFactory as SF, ModelAdapter, Task
from utils import load_model, get_dataloader
from utils.dataset_utils import get_transformation


def resolve_repository_path(
        repository: str | None,
        project_root: Path
) -> str:
    """
    Resolve a model or dataset repository path to an absolute path (beginning at the home directory)
    """
    if repository is None:
        raise ValueError("The repository path is required.")

    path = Path(repository).expanduser()

    if not path.is_absolute():
        path = project_root / path

    return str(path.resolve())


def run_benchmark(
        options: BenchmarkOptionConfig,
        models: List[ModelInfo],
        datasets: List[DatasetInfo],
        attacks: Optional[List[dict]] = None,
        metrics: Optional[List[dict]] = None,
        log: Optional[Logger] = None,
) -> list[ModelReportProps]:
    """
    This function take as input a full benchmark configuration and execute the benchmark.
    If the metrics and the attacks are not selected, then, by default, all the available attacks and metrics will be used.
    """
    #################################### 1. Valid Items ####################################
    ##### 1.1 Datasets
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

    ##### 1.2 Models
    # Filtering the attacks
    attacks: list[dict] = [
        attack
        for attack in attacks
        if attack.get("id", None) in AF.get_list_classes()
    ]
    if not any(attack.get("id") == "identitybaseline" for attack in attacks):
        attacks.append({"id": "identitybaseline"})
    #######################################################

    #################################### 2. Prepare Execution ####################################
    # 2.1 - Generate a unique id under which run all benchmark operations
    benchmark_id: str = datetime.now().strftime("%Y%m%dT%H%M%S")
    # 2.2 - create a single dict element with all necessary information to execute operation and merge end result.

    # Define an execution strategy for the benchmark at hand i.e. create an executor instance
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() and options.gpu else "cpu")
    output_path: str = options.output_path + f"/{datetime.now().strftime('%Y%m%dT%H%M%S')}"

    executor = BenchmarkExecutor(
        verbose=options.verbose,
        benchmark_id=benchmark_id,
        root_path=output_path,
        use_ray=options.use_ray,
    )
    list_reports: list[ModelReportProps] = []
    for model_cnf in models:
        task: Task = model_cnf.task if isinstance(model_cnf.task, Task) else Task.from_str(model_cnf.task)
        model: ModelAdapter = load_model(
            model_id=model_cnf.id or model_cnf.name,
            model_type=model_cnf.type,
            model_path=model_cnf.repository,
            api_url=model_cnf.api,
            task=task,
            device=device
        )
        transform = get_transformation(transformation=model_cnf.transformation)

        for dataset_cnf in datasets:
            if dataset_cnf.repository is None:
                raise ValueError("No dataset to load.")
            dataloader: DataLoader = get_dataloader(
                dataset_path=dataset_cnf.repository,
                batch=dataset_cnf.batch_size,
                subset=options.subset,
                transform=transform,
                num_workers=dataset_cnf.num_workers,
                name=dataset_cnf.name
            )
            #################### Defining the Statistic Composer ####################
            num_classes = model_cnf.num_classes
            if num_classes is None:
                batch, _ = next(iter(dataloader))
                out = model(batch.to(device))
                num_classes = out.shape[-1]

            if metrics is None or len(metrics) == 0:
                metrics = [{"id": metric} for metric in SF.get_list_classes(task={task})]
            metrics: list[dict] = [
                {
                    **metric,
                    "model": model,
                    "device": options.gpu,
                    "num_classes": num_classes,
                }
                for metric in metrics if metric.get("id") in SF.get_list_classes(task={task})
            ]
            statistics_composer = StatisticComposer(
                statistics=metrics,
                device=device
            )

            # 3.1 Start execution
            results: dict[str, ReportAttackProps] = executor.execute_jobs(
                model=model,
                dataloader=dataloader,
                attacks=attacks,
                statistics=statistics_composer,
                device=device,
                output_path=output_path,
                save_variables=options.variables_to_save,
                max_saved_elements=options.max_saved_elements,
            )
            global_metrics: dict = statistics_composer.compute_aggregator()

            # Global metrics
            identity: ReportAttackProps = results.pop("identitybaseline")
            # removing the metrics that I do not want because they refer to the attack's performance
            metrics: dict = identity.metrics.model_dump(
                exclude={
                    "misclassification",
                    "num_queries",
                    "robustness",
                    "risk",
                    "power"
                })
            metrics["num_samples"]: int = len(dataloader.dataset)
            global_metrics.update(metrics)
            # Here, for sure, the results dictionary does not have the "identity baseline" key
            model_report = ModelReportProps(
                info=model_cnf,
                metrics=ReportMetricsProps.model_validate(global_metrics),
                attacks=results,
            )
            list_reports.append(model_report)
            output_path: Path = Path(
                output_path).expanduser().resolve() / f"{model_cnf.id}/{dataset_cnf.id}"
            output_path.mkdir(parents=True, exist_ok=True)
            with open(output_path / "report.json", "a") as f:
                json.dump(model_report.model_dump(), f)
            ######### saving the results #########
            if log:
                log.info(
                    "Prepared %d job(s): %d model(s) x %d dataset(s) x %d attack(s).",
                    len(attacks), len(models), len(datasets), len(attacks),
                )

    return list_reports
