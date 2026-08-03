import json
import os.path
from datetime import datetime
from logging import Logger
from pathlib import Path
from typing import List, Optional

import torch
from torch.utils.data import DataLoader

from benchmarking.executor import BenchmarkExecutor
from models import BenchmarkOptionConfig, ModelInfo, DatasetInfo, ModelReportProps
from models.reports import ReportMetricsProps, ReportAttackProps
from nn_trust import AttackFactory as AF, StatisticComposer, StatisticsFactory as SF, ModelAdapter
from report import AdversarialReportGenerator
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
        models: List[ModelInfo],
        datasets: List[DatasetInfo],
        attacks: List[dict],
        metrics: List[dict],
        options: BenchmarkOptionConfig,
        log: Optional[Logger] = None,
) -> list[ModelReportProps]:
    """
    This function take as input a full benchmark configuration and execute the benchmark.
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
    if "identitybaseline" not in attacks:
        attacks.append({"id": "identitybaseline"})
    if len(attacks) == 0:
        raise ValueError("No proper attacks have been found.")
    #######################################################

    #################################### 2. Prepare Execution ####################################
    # 2.1 - Generate a unique id under which run all benchmark operations
    benchmark_id: str = datetime.now().strftime("%Y%m%dT%H%M%S")
    # 2.2 - create a single dict element with all necessary information to execute operation and merge end result.

    # Define an execution strategy for the benchmark at hand i.e. create an executor instance
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() and options.gpu else "cpu")
    executor = BenchmarkExecutor(
        verbose=options.verbose,
        benchmark_id=benchmark_id,
        root_path=options.output_path,
        use_ray=options.use_ray,
    )
    list_reports: list[ModelReportProps] = []
    for model_cnf in models:
        model: ModelAdapter = load_model(
            model_id=model_cnf.id or model_cnf.name,
            model_type=model_cnf.type,
            model_path=model_cnf.repository,
            api_url=model_cnf.api,
            task=model_cnf.task,
            device=device
        )
        transform = get_transformation(transformation=model_cnf.transformation)

        for dataset_cnf in datasets:
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

            metrics: list[dict] = [
                {
                    **metric,
                    "model": model,
                    "device": options.gpu,
                    "num_classes": num_classes,
                }
                for metric in metrics if metric.get("id") in SF.get_list_classes()
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
                device=device
            )

            global_metrics: dict = statistics_composer.compute_aggregator()

            if "identitybaseline" in results.keys():
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
                global_metrics.update(metrics)

            # Here, for sure, the results dictionary does not have the "identity baseline" key
            model_report = ModelReportProps(
                info=model_cnf,
                metrics=ReportMetricsProps.model_validate(global_metrics),
                attacks=results,
            )
            list_reports.append(model_report)
            output_path: Path = Path(
                options.output_path).expanduser().resolve() / f"{benchmark_id}/{model_cnf.id}_{dataset_cnf.id}"
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
