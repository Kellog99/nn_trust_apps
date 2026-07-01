import copy
import os
from datetime import datetime
from pathlib import Path

import ray

from benchmarking import BenchmarkConfig
from benchmarking import postprocess_results, create_benchmark_report
from benchmarking.benchmark_utils.execution import LocalRayExecutor, LocalSerialExecutor
from nn_trust.attack import AttackFactory as EAF


def run_benchmark_with_configuration(
        config: BenchmarkConfig,
        verbose=True
):
    """
    This function take as input a full benchmark configuration and execute the benchmark.
    """
    #################################### 1. Valid Configuration ####################################
    for dataset in config.datasets:
        if not Path(dataset.source_path).exists():
            raise ValueError(f"Dataset source path {dataset.source_path} does not exist.")
        for model in config.models:
            if "model_path" in model and not Path(model.model_path).exists():
                raise ValueError(f"Model path {model.model_path} does not exist.")
        for attack in config.attacks:
            EAF.get_info(attack.id)
    ################################################################################################

    #################################### 2. Prepare Execution ####################################
    # 2.1 - Generate a unique id under which run all benchmark operations
    benchmark_id: str = datetime.now().strftime("%Y%m%dT%H%M%S")
    # 2.2 - create a single dict element with all necessary information to execute operation and merge end result.
    inflated_configuration = []
    for model in config.models:
        # Get the idea from the model's information
        model_identification = next(mid for mid in [model.name, model.model_path.split("/")[-1]] if mid is not None)
        for dataset in config.datasets:
            for attack in config.attacks:
                atk_identification = next(atk_id for atk_id in [attack.name, attack.id] if atk_id is not None)
                inflated_configuration.append(copy.deepcopy({
                    "dataset": dataset,
                    "model": model,
                    "attack": attack,
                    "evaluation": config.evaluation,
                    "options": config.options,
                    "benchmark_info": {
                        "benchmark_id": benchmark_id,
                        "dataset_id": dataset.name,
                        "model_id": model_identification,
                        "atk_id": atk_identification,
                        "user_id": os.getlogin(),
                        "host": os.uname().nodename
                    }
                }))

    ##############################################################################################

    #################################### 3. Execution Strategy ####################################
    # Define an execution strategy for the benchmark at hand i.e. create an executor instance
    match config.options.mode:
        case "local_ray":
            ray.init()
            executor = LocalRayExecutor(root_path=config.options.output_path)
            if verbose:
                print("Using ray with cluster configuration:")
                print(ray.cluster_resources())
        case "local_serial":
            executor = LocalSerialExecutor(root_path=config.options.output_path)
        case _:
            raise ValueError(f"Execution mode '{config.options.mode}' is not supported.")
    print(f"Created executor instance\n{executor}")
    # 3.1 Start execution
    results = executor.execute_jobs(inflated_configuration)
    ###############################################################################################

    #################################### 3. Aggregate Results ####################################
    # Get output path from results and aggregate statistics for single attacks, for each dataset and model
    postprocess_results(results["output_path"])
    ###############################################################################################

    # 5 Optionally iterate of completed benchmark folders (postprocessed) and create a pdf report file.
    if config.options.output_format == "report":
        for dataset_and_model_dir in Path(results["output_path"]).iterdir():
            if dataset_and_model_dir.is_dir():
                print(f"Generating report for {dataset_and_model_dir.name}")
                create_benchmark_report(
                    dataset_and_model_dir=dataset_and_model_dir,
                    filename=dataset_and_model_dir / "report.pdf",
                    generated_by="Leonardo S.p.A.",
                    output_mode="pdf"
                )
