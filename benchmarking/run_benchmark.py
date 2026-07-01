from pathlib import Path

import ray

from benchmarking import BenchmarkConfig
from benchmarking.benchmark_utils.execution import LocalRayExecutor, LocalSerialExecutor
from benchmarking import validate_configuration, generate_benchmark_id, inflate_configuration, \
    postprocess_benchmark_run_resultsV2, create_benchmark_report


def run_benchmark_with_configuration(
        config: BenchmarkConfig,
        verbose=True
):
    """
    This function take as input a full benchmark configuration and execute the benchmark.
    """
    #########################################################################################
    # 1) - validate input configuration
    for dataset in config.datasets:
        if not Path(dataset.source_path).exists():
            raise ValueError(f"Dataset source path {dataset.source_path} does not exist.")
        for model in config.models:
            if "model_path" in model and not Path(model.model_path).exists():
                raise ValueError(f"Model path {model.model_path} does not exist.")
        for attack in config["attacks"]:
            class_id = attack
    #########################################################################################
    validate_configuration(benchmark_config=config)
    # 2 - if input configuration is a valida configuration prepare for execution
    # 2.1 - Generate a unique id under which run all benchmark operations
    benchmark_id = generate_benchmark_id()
    # 2.2 - Inflate input configuration into long for: Config([datasets], [models], [attacks]) -> [(benchmark_id, dataset_i, model_i, attack_i, ...), ...]
    inflated_configuration = inflate_configuration(benchmark_config=config, benchmark_id=benchmark_id)
    # 3 Define an execution strategy for the benchmark at hand i.e. create an executor instance
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
    results = executor(inflated_configuration)

    # 4 Get output path from results and aggregate statistics for single attacks, for each dataset - model pair
    postprocess_benchmark_run_resultsV2(results["output_path"])

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
