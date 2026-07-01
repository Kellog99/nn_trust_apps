import copy
import json
import os
import traceback
from datetime import datetime
from pathlib import Path

import yaml

from nn_trust.evaluation.composer import StatisticComposer, ConfigStatisticComposer
from .evaluation_utils import read_results_from_disk, aggregate_attacks_statistics
from .pydantic_models import BenchmarkConfig


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
    if root == "default":
        raise Exception("Env varible DATASETS_REPO must be specified.")
    if os.path.isabs(path):
        return Exception(
            "When performing benchmarking multi node or from attack server, dataset paths must be relative.")
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
    config_files = [f for f in os.listdir(config_dir) if
                    (f.endswith(".yaml") or f.endswith(".yml")) and f.startswith("config")]
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


def postprocess_results(benchmark_run_dir: str | Path, verbose=True):
    """
    This function has the role to process the results that come from the benchmark procedure.

    """
    benchmark_run_dir = Path(benchmark_run_dir).expanduser()

    for data_model_dir in [x for x in benchmark_run_dir.iterdir() if x.is_dir()]:
        try:
            results = read_results_from_disk(data_model_dir)
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
