import base64
import json
import logging
import os
import pathlib
import time
import traceback
from datetime import datetime
from pathlib import Path

from nn_trust.evaluation.composer import ConfigStatisticComposer, StatisticComposer

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
    from benchmark_utils.report_functions import (
    collect_dataset_aggregates_with_info, 
    enrich_with_ranks, extract_rank_metrics, 
    get_attacks_info, 
    transform_to_benchmark
    )
    from benchmark_utils.pdf_report import AdversarialReportGenerator
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
    from .benchmark_utils.report_functions import (
    collect_dataset_aggregates_with_info, 
    enrich_with_ranks, extract_rank_metrics, 
    get_attacks_info, 
    transform_to_benchmark
    )
    from .benchmark_utils.pdf_report import AdversarialReportGenerator
import numpy as np

# --- Timing --- #
SERIAL_TIME = None
PARALLEL_TIME = None
SERIAL_TIMES = []
PARALLEL_TIMES = []

def compute_time_benchmark_metrics(t_wall_serial: float,
                                   t_wall_parallel: float,
                                   num_workers: int,
                                   num_tasks: int,
                                   verbose: bool = True,
                                   output_file: str = "benchmark_results.txt"):
    """Compute and report time benchmark metrics."""

    speedup = t_wall_serial / t_wall_parallel
    efficiency = speedup / num_workers
    throughput = num_tasks / t_wall_parallel
    

    # Construct result dictionary
    results = {
        "Serial Time (s)": t_wall_serial,
        "Parallel Time (s)": t_wall_parallel,
        "Workers": num_workers,
        "Tasks": num_tasks,
        "Speedup": speedup,
        "Efficiency": efficiency,
        "Throughput (tasks/s)": throughput,
    }

    # Pretty print
    if verbose:
        print("\n====== Benchmark Metrics ======")
        for k, v in results.items():
            print(f"{k:35}: {v:.6f}" if isinstance(v, float) else f"{k:35}: {v}")
        print("================================\n")
    else:
        print(results)

    # Dump to text file
    with open(output_file, "w") as f:
        f.write("Benchmark Metrics\n")
        f.write("=================\n")
        for k, v in results.items():
            if isinstance(v, float):
                f.write(f"{k:35}: {v:.6f}\n")
            else:
                f.write(f"{k:35}: {v}\n")

    return results

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
                evaluator.evaluate_attacks()
                evaluator.save_results_to_disk(output_path=output_path)
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

def benchmark_multi_node_parallel(config: dict, executor : Executor):
    global SERIAL_TIME
    global PARALLEL_TIME
    global SERIAL_TIMES
    global PARALLEL_TIMES
    plans = []
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
                plans.append(plan)
                logging.warning(f"Evaluation results for {dataset["name"]}/{model_config["name"]} are saved to {output_path}")
            except Exception as e:
                logging.warning(f"\n\U0001F975 Evaluation of Model {model_config['name']} on Dataset {dataset['name']} failed with exception '{e}' +++\n")
                traceback.print_exc()
    
    if executor.num_actors>1:
        parallel_start = time.time()
    else:
        start = time.time()
    executor.execute_plan(plans, benchmark_id)
    if executor.num_actors>1:
        parallel_end = time.time()
        PARALLEL_TIME = parallel_end - parallel_start
        PARALLEL_TIMES.append(PARALLEL_TIME)
        if SERIAL_TIME:
            compute_time_benchmark_metrics(
                t_wall_serial=SERIAL_TIME,
                t_wall_parallel=PARALLEL_TIME,
                num_workers=executor.num_actors,
                num_tasks=len(config["datasets"]) * len(config["models"]) * len(config["attacks"]),
                verbose=True,
                output_file=output_path / "time_benchmark_metrics.txt"
            )
    else:
        end = time.time()
        SERIAL_TIME = end - start
        SERIAL_TIMES.append(SERIAL_TIME)
    structure = get_structure(output_path)
    with open(output_path / "structure.json", "w") as f:
        json.dump(structure, f)
    with open(output_path / "configuration.json", "w") as f:
        json.dump(config, f)
    return str(output_path)

def benchmark_single_node_parallel(config: dict,  executor : Executor):
    global SERIAL_TIME
    global PARALLEL_TIME
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
                plans.append(plan)
                logging.warning(f"Evaluation results for {dataset["name"]}/{model_config["name"]} are saved to {output_path}")
            except Exception as e:
                logging.warning(f"\n\U0001F975 Evaluation of Model {model_config['name']} on Dataset {dataset['name']} failed with exception '{e}' +++\n")
                traceback.print_exc()

    if executor.num_actors>1:
        parallel_start = time.time()
    else:
        start = time.time()
    executor.execute_plan(plans, benchmark_id)
    if executor.num_actors>1 :
        parallel_end = time.time()
        PARALLEL_TIME = parallel_end - parallel_start
        if SERIAL_TIME:
            compute_time_benchmark_metrics(
                t_wall_serial=SERIAL_TIME,
                t_wall_parallel=PARALLEL_TIME,
                num_workers=executor.num_actors,
                num_tasks=len(config["datasets"]) * len(config["models"]) * len(config["attacks"]),
                verbose=True,
                output_file=output_path / "time_benchmark_metrics.txt"
            )
    else:
        end = time.time()
        SERIAL_TIME = end - start
    structure = get_structure(output_path)
    with open(output_path / "structure.json", "w") as f:
        json.dump(structure, f)
    with open(output_path / "configuration.json", "w") as f:
        json.dump(config, f)
    return str(output_path)

def benchmark_from_attack_server(config: dict, executor : Executor):
    plans = []
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
                plans.append(plan)
                logging.warning(f"Evaluation results for {dataset["name"]}/{model_config["name"]} are saved to {output_path}")
            except Exception as e:
                logging.warning(f"\n\U0001F975 Evaluation of Model {model_config['name']} on Dataset {dataset['name']} failed with exception '{e}' +++\n")
                traceback.print_exc()
    benchmark_id = executor.execute_plan(plans, benchmark_id)
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

def create_benchmark_report(benchmark_run_dir: str | pathlib.Path, 
                            model_name : str, 
                            dataset_name : str, 
                            output_path : str = os.getcwd(),
                            pdf_report: bool = False):
    
    model_dir = os.path.join(benchmark_run_dir, dataset_name, model_name)

    with open(os.path.join(model_dir,'info.json'), "r", encoding="utf-8") as f:
                info = json.load(f)

    parts = benchmark_run_dir.split(os.sep)
    parent_benchmark_dir = os.sep.join(parts[:-1])  
    with open(os.path.join(model_dir,'aggregate_statistics.json'), "r", encoding="utf-8") as f:
        aggregate = json.load(f)
        aggregate["params"] = info["parameters"]
        results = collect_dataset_aggregates_with_info( #TODO: adapt this
            base_dir=parent_benchmark_dir,
            dataset=dataset_name,
            keep_latest_only=False,
        )

        out = transform_to_benchmark(results,task="classification") #TODO: adapt this
        out = enrich_with_ranks(out) #TODO: adapt this
        out = extract_rank_metrics(out,model_name) #TODO: adapt this
        num_b = len(results)
        out["total benchmarks"] = num_b
        aggregate = aggregate | out   

    statistics = {}
    for entry in os.listdir(model_dir):
        entry_path = os.path.join(model_dir, entry)
        if os.path.isdir(entry_path):
            stat_file = os.path.join(entry_path, "statistics.json")
            if os.path.exists(stat_file) and os.path.isfile(stat_file):
                try:
                    with open(stat_file, "r", encoding="utf-8") as sf:
                        sf_data = json.load(sf)
                        sf_data["name"] = get_attacks_info()[entry.lower()].name
                        sf_data["risk"] = 0.5
                        sf_data["num_queries"] = 1
                        sf_data["power"] = 0.5
                        statistics[entry.upper()] = sf_data
                except Exception as e:
                    logging.warning(f"Could not load statistics.json in '{entry_path}': {e}")
    
    report_data = {
        "info":info,
        "metrics": aggregate,
        "attacks": statistics
    }
    
    report_data["tool"] = "nntrust"
    report_data["dataset"] = dataset_name
    with open(os.path.join(output_path,"report.json"), "w", encoding="utf-8") as f:
                json.dump(report_data, f)
    if pdf_report==True:
        generator = AdversarialReportGenerator(logo_path='./resources/logo_leonardo.png')
        report_file = os.path.join(output_path,f"{dataset_name}_{model_name}.pdf")
        generator.generate(report_data, report_file)
        with open(report_file, 'rb') as pdf_file:
            pdf_bytes = pdf_file.read()
            return base64.b64encode(pdf_bytes).decode('utf-8')

# --- Main functions --- #
def benchmark(config: dict, executor: Executor):
    return benchmark_from_attack_server(config, executor=executor)

def benchmark_from_main(config: dict, mode : str, executor : Executor):
    if mode=="single_node_serial": 
        return benchmark_single_node_serial(config)
    elif mode=="single_node_parallel":
        return benchmark_single_node_parallel(config, executor=executor)
    elif mode=="multi_node_parallel":
        return benchmark_multi_node_parallel(config, executor=executor)
    else:
        raise ValueError(f"Benchmark mode {mode} not supported.")

def executors_factory(executor_type:str, num_workers: int = 1, num_gpus_per_worker: float = None) -> Executor:
    if executor_type=="ray":
        return RayActorPoolExecutor(num_actors=num_workers, num_gpus_per_actor=num_gpus_per_worker)
    else:
        raise ValueError(f"Executor type {executor_type} not supported.")

def main():
    handler = logging.StreamHandler()
    handler.addFilter(lambda record: record.name == "root")
    logging.basicConfig(level=logging.WARN, handlers=[handler])
    selected_config_path = config_file_path_selector(Path(__file__).parent / "config")
    config = read_config_file(config_filename=str(selected_config_path))
    config = BenchmarkConfig(**config)

   
    mode = config.options.mode
    num_workers = config.options.num_workers
    num_gpus_per_worker = config.options.num_gpus_per_worker
    executor_type = config.options.executor_type
    executor = executors_factory(executor_type=executor_type, num_workers=num_workers, num_gpus_per_worker=num_gpus_per_worker)
    executor.use_event_loop = False

    benchmark_from_main(config.model_dump(), 
                        mode=mode, 
                        executor=executor)
    
def time_benchmark():
    handler = logging.StreamHandler()
    handler.addFilter(lambda record: record.name == "root")
    logging.basicConfig(level=logging.WARN, handlers=[handler])
    print("------ Select serial configuration ------")
    selected_config_path = config_file_path_selector(Path(__file__).parent / "config")
    config = read_config_file(config_filename=str(selected_config_path))
    serial_config = BenchmarkConfig(**config)

    handler = logging.StreamHandler()
    handler.addFilter(lambda record: record.name == "root")
    logging.basicConfig(level=logging.WARN, handlers=[handler])
    print("------ Select parallel configuration ------")
    selected_config_path = config_file_path_selector(Path(__file__).parent / "config")
    config = read_config_file(config_filename=str(selected_config_path))
    parallel_config = BenchmarkConfig(**config)

    for _ in range(1):

        
        mode = serial_config.options.mode
        num_workers = serial_config.options.num_workers
        num_gpus_per_worker = serial_config.options.num_gpus_per_worker
        executor_type = serial_config.options.executor_type
        executor = executors_factory(executor_type=executor_type, num_workers=num_workers, num_gpus_per_worker=num_gpus_per_worker)
        executor.use_event_loop = False

        benchmark_from_main(serial_config.model_dump(), 
                            mode=mode, 
                            executor=executor)
        
        mode = parallel_config.mode
        num_workers = parallel_config.num_workers
        num_gpus_per_worker = parallel_config.num_gpus_per_worker
        executor_type = parallel_config.executor_type
        executor = executors_factory(executor_type=executor_type, num_workers=num_workers, num_gpus_per_worker=num_gpus_per_worker)
        executor.use_event_loop = False

        benchmark_from_main(parallel_config.model_dump(), 
                            mode=mode, 
                            executor=executor)
        
        print("=========================================")
        print("|                                       |")
        print("|                                       |")
        print("|                                       |")
        print("|                                       |")
        print("Number of runs: ", len(SERIAL_TIMES))
        print("Mean over serial times: ", np.mean(SERIAL_TIMES))
        print("Mean over parallel times: ", np.mean(PARALLEL_TIMES))
        print("|                                       |")
        print("|                                       |")
        print("|                                       |")
        print("|                                       |")
        print("=========================================")
   
def save():
    postprocess_benchmark_run_results("/home/cristiano-carta/Desktop/output/20251117T114235")
    create_benchmark_report(benchmark_run_dir="/home/cristiano-carta/Desktop/output/20251117T114235",
                            dataset_name="landanimals",
                            model_name="convnext_base.clip_laion2b_augreg_ft_in12k_in1k",
                            output_path="/home/cristiano-carta/Desktop/output",
                            pdf_report=True)

if __name__ == "__main__":
    main()
    #save()
    #time_benchmark()






