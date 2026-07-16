import argparse
import json
import logging
from pathlib import Path
from typing import TypeVar, Type

import yaml

from benchmarking import run_benchmark
from models import BenchmarkOptionConfig, ModelInfo, DatasetInfo
from nn_trust import AttackFactory as AF, Task, LossFactory as LF, StatisticsFactory as SF

logger = logging.getLogger("benchmark")

T = TypeVar("T", ModelInfo, DatasetInfo)


def load_info(entry: dict, info_cls: Type[T]) -> T:
    """
    Build a ModelInfo/DatasetInfo from one entry of the config's "models"/"datasets"
    list. Two shapes are supported:

      - {"source_path": "..."}                -> info is read from <source_path>/info.json,
                                                   repository defaults to source_path.
      - {<info fields directly inline>, ...}   -> validated as-is; "repository" must be set.
    """
    source_path = entry.get("source_path")
    if source_path:
        source_path = Path(source_path).expanduser()
        info_json = source_path / "info.json"

        if not info_json.parent.exists():
            raise FileNotFoundError(f"The model's folder, {info_json.parent}, does not exist.")
        if not info_json.exists():
            raise FileNotFoundError(f"Expected an info.json under {source_path}, found none.")

        with open(info_json, "rb") as f:
            data = json.load(f)
        info = info_cls.model_validate(data)

        if getattr(info, "repository", None) is None:
            info.repository = str(source_path)
        return info

    info = info_cls.model_validate(entry)
    if getattr(info, "repository", None) is None:
        raise ValueError(
            f"Entry {entry.get('id', entry.get('name', '<unknown>'))} needs either a "
            f"'source_path' or an explicit 'repository'."
        )
    return info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="Attack benchmark",
        description="Run adversarial-attack benchmarks defined in a YAML config file."
    )
    parser.add_argument(
        "--config_path",
        "-cnf",
        required=True,
        type=Path,
        help="Path to the Benchmark configuration file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(
        level="INFO",
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    config_path = args.config_path.expanduser()

    ########### Checking ig the configuration file exists ###########
    if not config_path.exists():
        raise FileNotFoundError(f"The path to the Benchmark configuration file does not exist: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    if not config:
        raise ValueError(f"Configuration file {config_path} is empty or could not be parsed.")

    ####################### 1) Getting the models #######################
    models = [load_info(entry, ModelInfo) for entry in config.get("models", [])]
    if not models:
        raise ValueError("The configuration must define at least one entry under 'models'.")
    #####################################################################

    ####################### 2) Getting the datasets #######################
    datasets = [load_info(entry, DatasetInfo) for entry in config.get("datasets", [])]
    if not datasets:
        raise ValueError("The configuration must define at least one entry under 'datasets'.")
    ######################################################################

    ####################### 3) Getting the attacks #######################
    # Models may have different tasks (e.g. classification + detection in the same
    # sweep) -- collect the union of tasks rather than assuming a single model's task.
    tasks = {Task.from_str(model.task) for model in models}

    available_attacks = AF.get_list_classes(task=tasks)
    requested_attacks = config.get("attacks", [])
    attacks = (
        [atk for atk in requested_attacks if atk.get("id", None) in available_attacks]
        if requested_attacks else available_attacks
    )
    if not attacks:
        raise ValueError("No registered attacks matched the requested config/task(s).")
    ######################################################################

    ####################### 4) Getting the metrics #######################
    available_metrics = SF.get_list_classes(task=tasks)
    requested_metrics = config.get("metrics", [])
    metrics: list[dict] = [mtr for mtr in requested_metrics if mtr.get("id", None) in available_metrics]

    if not metrics:
        raise ValueError("No registered metrics matched the requested config/task(s).")
    ######################################################################

    ####################### Setting the Benchmarking options #######################
    options = BenchmarkOptionConfig.model_validate(config.get("options", {}))

    logger.info(
        "Running benchmark:\n %d model(s),\n %d dataset(s),\n %d attack(s),\n %d metric(s)",
        len(models),
        len(datasets),
        len(attacks),
        len(metrics),
    )
    logger.info(
        "Output path: %s | Ray: %s | PDF report: %s",
        options.output_path, options.use_ray, options.create_pdf,
    )

    results = run_benchmark(
        models=models,
        datasets=datasets,
        attacks=attacks,
        metrics=metrics,
        options=options,
    )

    logger.info(
        "Done. %d succeeded, %d failed. Results: %s",
        len(results["succeeded"]), len(results["failed"]), results["output_path"],
    )
