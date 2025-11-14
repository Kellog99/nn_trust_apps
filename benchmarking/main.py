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
    from benchmark_utils.executor import RayActorPoolExecutor, Executor
    from benchmark_utils.utils import resolve_path
except ModuleNotFoundError:
    # when used from attack.server it need to import as if working as a module
    from .benchmark_utils import (
        read_config_file,
        BenchmarkConfig,
        Evaluator,
        get_structure,
        config_file_path_selector
    )
    from .benchmark_utils.executor import RayActorPoolExecutor, Executor
    from .benchmark_utils.utils import resolve_path


# --- Bencharking functions --- #
def benchmark_single_node_serial(config: dict):
    output_path = Path(config["options"]["output_path"])
    benchmark_id = datetime.now().strftime("%Y%m%dT%H%M%S")
    output_path = output_path / benchmark_id
    os.makedirs(output_path, exist_ok=False)
    config["output_path"] = str(output_path)
    logging.info(f"Benchmark run will be save to {output_path}")
    for dataset_id, dataset in enumerate(config["datasets"]):
        for model_id, model_config in enumerate(config["models"]):
            try:
                evaluator = Evaluator.from_config(config=config, dataset=dataset, model_config=model_config)
                results = evaluator.evaluate_attacks()
                logging.warning(f"Evaluation results for {dataset["name"]}/{model_config["name"]} are saved to {output_path}")
                return results
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

def benchmark_multi_node_parallel(config: dict, executor : Executor, num_actors: int = 1, num_gpus_per_actor: int = None):
    output_path = Path(config["options"]["output_path"])
    benchmark_id = datetime.now().strftime("%Y%m%dT%H%M%S")
    output_path = output_path / benchmark_id
    os.makedirs(output_path, exist_ok=False)
    config["output_path"] = str(output_path)
    logging.info(f"Benchmark run will be save to {output_path}")
    for dataset_id, dataset in enumerate(config["datasets"]):
        dataset["relative_source_path"] = dataset["source_path"]
        path = resolve_path(dataset["source_path"])
        dataset["source_path"] = path
        for model_id, model_config in enumerate(config["models"]):
            try:
                evaluator = Evaluator.from_config(config=config, dataset=dataset, model_config=model_config)
                plan = evaluator.plan_attacks_evaluation()
                if num_gpus_per_actor is not None:
                    os.environ["FRACTION_FOR_GPU_ACTOR"]=str(num_gpus_per_actor)
                executor.execute_plan(plan, benchmark_id)
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

def benchmark_single_node_parallel(config: dict,  executor : Executor, num_actors: int = 1, num_gpus_per_actor: int = None):
    plans = []
    output_path = Path(config["options"]["output_path"])
    benchmark_id = datetime.now().strftime("%Y%m%dT%H%M%S")
    output_path = output_path / benchmark_id
    os.makedirs(output_path, exist_ok=False)
    config["output_path"] = str(output_path)
    logging.info(f"Benchmark run will be save to {output_path}")
    for dataset_id, dataset in enumerate(config["datasets"]):
        dataset["relative_source_path"] = dataset["source_path"]
        if not os.path.isabs(Path(dataset["source_path"]).resolve()):
            raise Exception(f"Dataset path {dataset['source_path']} is not absolute. Please provide absolute paths for single-node benchmarks.")
        for model_id, model_config in enumerate(config["models"]):
            try:
                evaluator = Evaluator.from_config(config=config, dataset=dataset, model_config=model_config)
                plan = evaluator.plan_attacks_evaluation()
                if num_gpus_per_actor is not None:
                    os.environ["FRACTION_FOR_GPU_ACTOR"]=str(num_gpus_per_actor)
                plans.append(plan)
                logging.warning(f"Evaluation results for {dataset["name"]}/{model_config["name"]} are saved to {output_path}")
            except Exception as e:
                logging.warning(f"\n\U0001F975 Evaluation of Model {model_config['name']} on Dataset {dataset['name']} failed with exception '{e}' +++\n")
                traceback.print_exc()
    executor.execute_plan(plans, benchmark_id)
    structure = get_structure(output_path)
    with open(output_path / "structure.json", "w") as f:
        json.dump(structure, f)
    with open(output_path / "configuration.json", "w") as f:
        json.dump(config, f)
    return str(output_path)

def benchmark_from_attack_server(config: dict, executor):
    output_path = Path(config["options"]["output_path"])
    os.environ["BENCHMARK_OUTPUT_DIR"] = str(output_path)
    benchmark_id = datetime.now().strftime("%Y%m%dT%H%M%S")
    output_path = output_path / benchmark_id
    os.makedirs(output_path, exist_ok=False)
    config["output_path"] = str(output_path)
    logging.info(f"Benchmark run will be save to {output_path}")
    for dataset_id, dataset in enumerate(config["datasets"]):
        dataset["relative_source_path"] = dataset["source_path"]
        path = resolve_path(dataset["source_path"])
        dataset["source_path"] = path
        for model_id, model_config in enumerate(config["models"]):
            try:
                evaluator = Evaluator.from_config(config=config, dataset=dataset, model_config=model_config)
                plan = evaluator.plan_attacks_evaluation()
                benchmark_id = executor.execute_plan(plan, benchmark_id)
                logging.warning(f"Evaluation results for {dataset["name"]}/{model_config["name"]} are saved to {output_path}")
            except Exception as e:
                logging.warning(f"\n\U0001F975 Evaluation of Model {model_config['name']} on Dataset {dataset['name']} failed with exception '{e}' +++\n")
                traceback.print_exc()

    structure = get_structure(output_path)
    with open(output_path / "structure.json", "w") as f:
        json.dump(structure, f)
    with open(output_path / "configuration.json", "w") as f:
        json.dump(config, f)
    return benchmark_id


# --- Aggregation and pdf report --- #
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


def benchmark(config: dict, executor: Executor):
    return benchmark_from_attack_server(config, executor=executor)

def benchmark_from_main(config: dict, mode : str, executor : Executor, num_actors: int = 1, num_gpus_per_actor: int = None):
    if mode=="single_node_serial": 
        return benchmark_single_node_serial(config)
    elif mode=="single_node_parallel":
        return benchmark_single_node_parallel(config, executor=executor, num_actors=num_actors, num_gpus_per_actor=num_gpus_per_actor)
    elif mode=="multi_node_parallel":
        return benchmark_multi_node_parallel(config, executor=executor, num_actors=num_actors, num_gpus_per_actor=num_gpus_per_actor)
    else:
        raise ValueError(f"Benchmark mode {mode} not supported.")

def executors_factory(executor_type:str, num_workers: int = 1) -> Executor:
    if executor_type=="ray":
        return RayActorPoolExecutor(num_actors=num_workers)
    else:
        raise ValueError(f"Executor type {executor_type} not supported.")

def main():
    handler = logging.StreamHandler()
    handler.addFilter(lambda record: record.name == "root")
    logging.basicConfig(level=logging.WARN, handlers=[handler])
    selected_config_path = config_file_path_selector(Path(__file__).parent / "config")
    config = read_config_file(config_filename=str(selected_config_path))
    config = BenchmarkConfig(**config)

    # --- TO ADD IN THE PARSER --- #
    mode = "single_node_parallel"
    num_workers = 2
    num_gpus_per_worker = 1
    executor_type = "ray"
    executor = executors_factory(executor_type=executor_type, num_workers=num_workers)
    executor.use_event_loop = False
    # ---------------------------- #

    benchmark_from_main(config.model_dump(), 
                        mode=mode, 
                        executor=executor, 
                        num_actors=num_workers, 
                        num_gpus_per_actor=num_gpus_per_worker)

if __name__ == "__main__":
    main()






