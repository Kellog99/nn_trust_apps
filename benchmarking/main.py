import json
from pathlib import Path
import os
import logging
import traceback
from datetime import datetime
import pathlib
import ray

from nn_trust.attack.evaluation.composer import ConfigStatisticComposer, StatisticComposer
try:
    from benchmark_utils import (
        read_config_file,
        BenchmarkConfig,
        Evaluator,
        get_structure,
        config_file_path_selector
    )
except ModuleNotFoundError:
    # when used from attack.server it need to import as if working as a module
    from .benchmark_utils import (
        read_config_file,
        BenchmarkConfig,
        Evaluator,
        get_structure,
        config_file_path_selector
    )


def resolve_path(path: str) -> str:
    """
    Returns an absolute path. If the given path is not absolute,
    it prepends a root path read from the environment variable ROOT_PATH.
    """
    root = os.getenv("ROOT_PATH", "")
    if os.path.isabs(path):
        return path
    return os.path.join(root, path)

def benchmark_(config: dict):
    #TODO: handle env path to dataset folder
    #os.environ["ROOT_PATH"] = "/home/cristiano-carta/Desktop/datasets"
    os.environ["RAY_OVERRIDE_ENVIRONMENT_VARIABLES_ALLOWLIST"] = "*"
    output_path = Path(config["options"]["output_path"])
    output_path = output_path / datetime.now().strftime("%Y%m%dT%H%M%S")
    os.makedirs(output_path, exist_ok=False)
    config["output_path"] = str(output_path)
    logging.info(f"Benchmark run will be save to {output_path}")
    for dataset_id, dataset in enumerate(config["datasets"]):
        dataset["relative_source_path"] = dataset["source_path"]
        path = resolve_path(dataset["source_path"])
        dataset["source_path"] = path
        for model_id, model_config in enumerate(config["models"]):
            try:
                ray.init(ignore_reinit_error=True,runtime_env={
                    "py_modules": ["/home/cristiano-carta/Desktop/projects/nn_trust_apps/benchmarking"]
                })
                #TODO: fix imports
                try:
                    from benchmark_utils.executor import RayActorPoolExecutor
                except ModuleNotFoundError:
                    from .benchmark_utils.executor import RayActorPoolExecutor
            
                os.chdir("/home/cristiano-carta/Desktop/projects/nn_trust_apps")
                evaluator = Evaluator.from_config(config=config, dataset=dataset, model_config=model_config)
                plan = evaluator.plan_attacks_evaluation()
                executor = RayActorPoolExecutor(num_actors=2)
                executor.execute_plan(plan)
                
                logging.warning(f"Evaluation results for {dataset["name"]}/{model_config["name"]} are saved to {output_path}")
            except Exception as e:
                logging.warning(f"\n\U0001F975 Evaluation of Model {model_config['name']} on Dataset {dataset['name']} failed with exception '{e}' +++\n")
                traceback.print_exc()
                    # Saving the structure for the report

    structure = get_structure(output_path)
    with open(output_path / "structure.json", "w") as f:
        json.dump(structure, f)
    with open(output_path / "configuration.json", "w") as f:
        json.dump(config, f)
    return str(output_path)

def parallel_benchmark_(config: dict, executor):

    #TODO: handle env path to dataset folder
    #os.environ["ROOT_PATH"] = "/home/cristiano-carta/Desktop/datasets"
    os.environ["RAY_OVERRIDE_ENVIRONMENT_VARIABLES_ALLOWLIST"] = "*"
    output_path = Path(config["options"]["output_path"])
    output_path = output_path / datetime.now().strftime("%Y%m%dT%H%M%S")
    os.makedirs(output_path, exist_ok=False)
    config["output_path"] = str(output_path)
    logging.info(f"Benchmark run will be save to {output_path}")

    for dataset_id, dataset in enumerate(config["datasets"]):
        dataset["relative_source_path"] = dataset["source_path"]
        path = resolve_path(dataset["source_path"])
        dataset["source_path"] = path
        for model_id, model_config in enumerate(config["models"]):
            try:
                #TODO: fix imports
                ray.init(ignore_reinit_error=True,runtime_env={
                    "py_modules": ["/home/cristiano-carta/Desktop/projects/nn_trust_apps/benchmarking"]
                })
                try:
                    from benchmark_utils.executor import RayActorPoolExecutor
                except ModuleNotFoundError:
                    from .benchmark_utils.executor import RayActorPoolExecutor
            
                os.chdir("/home/cristiano-carta/Desktop/projects/nn_trust_apps")

                evaluator = Evaluator.from_config(config=config, dataset=dataset, model_config=model_config)
                plan = evaluator.plan_attacks_evaluation()
                #executor = RayActorPoolExecutor(num_actors=2)
                executor.execute_plan(plan)

                logging.warning(f"Evaluation results for {dataset["name"]}/{model_config["name"]} are saved to {output_path}")
            except Exception as e:
                logging.warning(f"\n\U0001F975 Evaluation of Model {model_config['name']} on Dataset {dataset['name']} failed with exception '{e}' +++\n")
                traceback.print_exc()

    structure = get_structure(output_path)
    with open(output_path / "structure.json", "w") as f:
        json.dump(structure, f)
    with open(output_path / "configuration.json", "w") as f:
        json.dump(config, f)
    return str(output_path)


def postprocess_benchmark_run_results(benchmark_run_dir: str | pathlib.Path, verbose=True):
    """Iterate over different datasets and models, and where possible apply statistics aggregation"""
    benchmark_run_dir = Path(benchmark_run_dir)
    with open(benchmark_run_dir / "configuration.json", "r") as fconfiguration:
        config = json.load(fconfiguration)
    datasets_dir = [dataset_dir for dataset_dir in benchmark_run_dir.iterdir() if dataset_dir.is_dir()]
    for dataset_dir in datasets_dir:
        models_dir = [model_dir for model_dir in dataset_dir.iterdir() if model_dir.is_dir()]
        for model_dir in models_dir:
            try:
                with open(model_dir / "info.json", "r") as finfo:
                    info = json.load(finfo)
                statistics_composer = StatisticComposer(config=ConfigStatisticComposer(
                    statistics=config["evaluation"]["statistics"],
                    num_classes=info["classes"],
                ))
                statistics_composer.aggregator()
                results = Evaluator.read_results_from_disk(model_dir)
                aggregate_statistics = Evaluator.aggregate_attacks_statistics(
                    statistics_composer=statistics_composer,
                    results=results
                )
                with open(model_dir / "aggregate_statistics.json", "w") as fagg_statistics:
                    json.dump(aggregate_statistics, fagg_statistics)
            except Exception as e:
                if verbose:
                    print(f"Failed postprocessing on {dataset_dir.name}/{model_dir.name}")
                    print(e)
                    traceback.print_exc()


def benchmark(config: dict, executor=None):
    output_path = parallel_benchmark_(config, executor=executor)
    #postprocess_benchmark_run_results(output_path)


def main():
    handler = logging.StreamHandler()
    handler.addFilter(lambda record: record.name == "root")
    logging.basicConfig(level=logging.WARN, handlers=[handler])
    # get the parser
    selected_config_path = config_file_path_selector(Path(__file__).parent / "config")
    config = read_config_file(config_filename=str(selected_config_path))
    config = BenchmarkConfig(**config)

    output_path = benchmark(config.model_dump())
    #print(f"Results saved to {output_path}")
    #postprocess_benchmark_run_results(output_path)

if __name__ == "__main__":
    main()






