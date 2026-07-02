import os
from datetime import datetime
from pathlib import Path
from typing import List

from benchmarking import postprocess_results, create_benchmark_report
from benchmarking.executor import BenchmarkExecutor
from models import BenchmarkOptionConfig, ModelInfo, DatasetInfo, RegisteredObject
from nn_trust.attack import AttackFactory as EAF


def run_benchmark(
        models: List[ModelInfo],
        datasets: List[DatasetInfo],
        attacks: List[RegisteredObject],
        metrics: List[RegisteredObject],
        options: BenchmarkOptionConfig,
) -> dict:
    """
    This function take as input a full benchmark configuration and execute the benchmark.
    """
    #################################### 1. Valid Dataset ####################################
    for dataset in datasets:
        if not Path(dataset.repository).exists():
            raise ValueError(f"Dataset source path {dataset.repository} does not exist.")
        for model in models:
            if not Path(model.repository).exists():
                raise ValueError(f"Model path {model.repository} does not exist.")
        for attack in attacks:
            EAF.get_info(attack.id)
    ##########################################################################################

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
    postprocess_results(results["output_path"])
    ###############################################################################################

    #################################### 5. PDF generation ####################################
    # Optionally, after the benchmark, it could be created the PDF report of the vulnerabilities
    if options.create_pdf:
        for dataset_and_model_dir in Path(results["output_path"]).iterdir():
            if dataset_and_model_dir.is_dir():
                print(f"Generating report for {dataset_and_model_dir.name}")
                create_benchmark_report(
                    dataset_and_model_dir=dataset_and_model_dir,
                    filename=dataset_and_model_dir / "report.pdf",
                    generated_by="Leonardo S.p.A.",
                    output_mode="pdf"
                )
    ###########################################################################################

    return results
