import os
import copy
import yaml
from pathlib import Path
import json
from datetime import datetime
import traceback

import ray

from nn_trust.evaluation.composer import StatisticComposer, ConfigStatisticComposer
from .evaluation_utils import read_results_from_diskV2, aggregate_attacks_statistics
from .pdf_report import create_benchmark_report
from .pydantic_models import BenchmarkConfig
from .execution import LocalRayExecutor, LocalSerialExecutor


def read_config_file(config_filename: str) -> BenchmarkConfig:
    """
    Read the configuration file and return the content as a dictionary.
    """
    if not os.path.exists(config_filename):
        raise ValueError(f"File not found: {config_filename}")

    if config_filename.endswith((".yaml", ".yml")):
        with open(config_filename, "r") as f:
            config_data = yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported file format: {config_filename}")
    
    return BenchmarkConfig(**config_data)

def resolve_path(path: str) -> str:
    """
    Returns an absolute path. If the given path is not absolute,
    it prepends a root path read from the environment variable ROOT_PATH.
    """
    root = os.getenv("DATASETS_REPO", "default")
    if root=="default":
        raise Exception("Env varible DATASETS_REPO must be specified.")
    if os.path.isabs(path):
        return Exception("When performing benchmarking multi node or from attack server, dataset paths must be relative.")
    return os.path.join(root, path)





def get_structure(path: Path | str) -> dict:
    path = path if isinstance(path, Path) else Path(path)
    if path.is_dir():
        out = {element: get_structure(path=path / element) for element in os.listdir(path) if (path / element).is_dir()}
        out["files"] = [element for element in os.listdir(path) if not (path / element).is_dir()]

        return out
    else:
        return {"files": path.name}




def config_file_path_selector(config_dir: Path | str = ".") -> Path:
    """Seletc a YAML configuration file from the script directory."""
    config_files = [f for f in os.listdir(config_dir) if (f.endswith(".yaml") or f.endswith(".yml")) and f.startswith("config")]
    if not config_files:
        raise FileNotFoundError("No YAML configuration files found in the script directory.")
    print("Available configuration files:")
    for idx, fname in enumerate(config_files):
        print(f"{idx}: {fname}")
    selected_idx = int(input("Select configuration file by index: "))
    if selected_idx < 0 or selected_idx >= len(config_files):
        raise IndexError("Selected index is out of range.")
    selected_config_path = config_dir / config_files[selected_idx]
    return selected_config_path



def postprocess_benchmark_run_resultsV2(benchmark_run_dir: str | Path, verbose=True):
    benchmark_run_dir = Path(benchmark_run_dir)

    for data_model_dir in [x for x in benchmark_run_dir.iterdir() if x.is_dir()]:
        try:
            results = read_results_from_diskV2(data_model_dir)
            info = results["info"]
            with open(data_model_dir / "info.json", "w") as f:
                json.dump(info, f, indent=4)

            statistics_composer = StatisticComposer(config=ConfigStatisticComposer(
                statistics=info["statistics"],
                num_classes=info["model_info"]["num_classes"],
            ))
            statistics_composer.aggregator()

            aggregate_statistics = aggregate_attacks_statistics(
                statistics_composer=statistics_composer,
                results=results
            )

            with open(data_model_dir / "aggregate_statistics.json", "w") as fagg_statistics:
                json.dump(aggregate_statistics, fagg_statistics)

            if verbose:
                print(f"Postprocessed {data_model_dir.relative_to(benchmark_run_dir)}")
        except Exception as e:
            print(f"Failed postprocessing on {data_model_dir.relative_to(benchmark_run_dir)}")
            print(e)
            traceback.print_exc()

def validate_configuration(benchmark_config: BenchmarkConfig):
    # validate configuration (test if model, dataset and attacks all make sense, if paths exists, etc.)
    config = benchmark_config.model_dump()
    for dataset in config["datasets"]:
        if not Path(dataset['source_path']).exists():
            raise ValueError(f"Dataset source path {dataset['source_path']} does not exist.")
    for model in config["models"]:
        if "model_path" in model and not Path(model["model_path"]).exists():
            raise ValueError(f"Model path {model['model_path']} does not exist.")
    for attack in config["attacks"]:
        class_id = attack

def inflate_configuration(benchmark_config: BenchmarkConfig, benchmark_id: str) -> list[dict]:
    # Inflate configuration
    # For each attack, model and attack, create a single dict element with all necessary information to execute operation and merge end result.
    # Create product from (dataset, model, attack) and add evaluation and options to each element, that are constant.
    # Inject benchmark idenfication element for dretrieval and merging results at the end.
    # Returns:
    # List of attack execution configurations sorted in order of MODEL > DATASET > ATTACKS
    config = benchmark_config.model_dump()
    inflated_configuration = []
    for i, model in enumerate(config["models"]):
        model_identifications = [model.get("name"), model.get("id"), model.get("model_path").split("/")[-1]]
        model_identification = next(mid for mid in model_identifications if mid is not None)
        for j, dataset in enumerate(config["datasets"]):       
            for k, attack in enumerate(config["attacks"]):
                atk_identifications = [attack.get("name"), attack.get("id")]
                atk_identification = next(atk_id for atk_id in atk_identifications if atk_id is not None)
                inflated_configuration.append(copy.deepcopy({
                    "dataset": dataset,
                    "model": model,
                    "attack": attack,
                    "evaluation": config["evaluation"],
                    "options": config["options"],
                    "benchmark_info" : {
                        "benchmark_id":benchmark_id,
                        "dataset_id":dataset.get("name"),
                        "model_id":model_identification,
                        "atk_id":atk_identification,
                        "user_id":os.getlogin(),
                        "host":os.uname().nodename
                    }
                }))
    return inflated_configuration

def generate_benchmark_id():
    """Generate a unique benchmark id based on current timestamp. Format: YYYYMMDDTHHMMSS"""
    return datetime.now().strftime("%Y%m%dT%H%M%S")


def run_benchmark_with_configuration(config: BenchmarkConfig, verbose=True):
    """This function take as input a full benchmark configuration and execute the benchmark.
    """
    # 1 - validate input configuration
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