import os
from datetime import datetime
from pathlib import Path
from typing import List

import ray

from benchmarking.executor import BenchmarkExecutor
from models import BenchmarkOptionConfig, ModelInfo, DatasetInfo, RegisteredObject
from nn_trust.attack import AttackFactory as EAF
from report import AdversarialReportGenerator


# Resolve a model or dataset repository path to an absolute path (beginning at the home directory)
def resolve_repository_path(repository: str | None, project_root: Path) -> str:
    if repository is None:
        raise ValueError("The repository path is required.")

    path = Path(repository).expanduser()

    if not path.is_absolute():
        path = project_root / path

    return str(path.resolve())


def run_benchmark(
        models: List[ModelInfo],
        datasets: List[DatasetInfo],
        attacks: List[RegisteredObject],
        metrics: List[str],
        options: BenchmarkOptionConfig,
) -> dict:
    """
    This function take as input a full benchmark configuration and execute the benchmark.
    """
    #################################### 1. Valid Dataset ####################################
    for dataset in datasets:
        if dataset.repository is None:
            raise ValueError("The path to the dataset repository is required.")
        if not Path(dataset.repository).exists():
            raise FileNotFoundError(f"Dataset source path {dataset.repository} does not exist.")
    for model in models:
        if model.repository is None:
            raise ValueError("The path to the model repository is required.")
        if not Path(model.repository).exists():
            raise FileNotFoundError(f"Model path {model.repository} does not exist.")
    for attack in attacks:
        # checks that attack is registered
        EAF.get_info(attack.id)
    ##########################################################################################

    # Resolve model and dataset repositories before building jobs so local execution and Ray workers receive absolute paths independent of their working directory
    project_root = Path(__file__).resolve().parents[1]

    datasets = [
        dataset.model_copy(
            update={
                "repository": resolve_repository_path(
                    dataset.repository,
                    project_root,
                )
            }
        )
        for dataset in datasets
    ]

    models = [
        model.model_copy(
            update={
                "repository": resolve_repository_path(
                    model.repository,
                    project_root,
                )
            }
        )
        for model in models
    ]

    # Expose the local nn_trust submodule on PYTHONPATH
    if options.use_ray and not ray.is_initialized():
        nn_trust_path = project_root / "submodules" / "nn_trust"

        ray.init(
            runtime_env={
                "working_dir": str(project_root),
                "env_vars": {
                    "PYTHONPATH": (
                        f"{project_root}:{nn_trust_path}:"
                        f"{os.environ.get('PYTHONPATH', '')}"
                    )
                },
            }
        )
    #################################### 2. Prepare Execution ####################################
    # 2.1 - Generate a unique id under which run all benchmark operations
    benchmark_id: str = datetime.now().strftime("%Y%m%dT%H%M%S")
    # 2.2 - create a single dict element with all necessary information to execute operation and merge end result.
    inflated_configuration = []
    for model in models:
        for dataset in datasets:
            for attack in attacks:
                inflated_configuration.append({
                    "dataset": dataset,
                    "model": model,
                    "attack": attack,
                    "evaluation": metrics,
                    "options": options,
                    "benchmark_info": {
                        "benchmark_id": benchmark_id,
                        "dataset_id": dataset.name,
                        "model_id": model.id,
                        "atk_id": attack.id,
                        "user_id": os.getlogin(),
                        "host": os.uname().nodename
                    }
                })
    ##############################################################################################

    #################################### 3. Execution Strategy ####################################
    # Define an execution strategy for the benchmark at hand i.e. create an executor instance
    executor = BenchmarkExecutor(
        root_path=options.output_path,
        use_ray=options.use_ray,
    )

    # 3.1 Start execution
    results = executor.execute_jobs(inflated_configuration)
    ###############################################################################################

    #################################### 4. Aggregate Results ####################################
    # Get output path from results and aggregate statistics for single attacks, for each dataset and model
    # removing do to critical issues
    #postprocess_results(results["output_path"])
    ###############################################################################################

    #################################### 5. PDF generation ####################################
    # Optionally, after the benchmark, it could be created the PDF report of the vulnerabilities
    if options.create_pdf:
        report = AdversarialReportGenerator()
        report.generate(
            data=results,
            output_path=options.output_path,
            header_logo_path=None,
        )
    ###########################################################################################

    return results
