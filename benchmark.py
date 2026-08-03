import argparse
import json
import logging
from pathlib import Path
from typing import TypeVar, Type

import yaml

from benchmarking import run_benchmark
from models import BenchmarkOptionConfig, ModelInfo, DatasetInfo, ModelReportProps
from nn_trust import AttackFactory as AF, Task, StatisticsFactory as SF
from report import AdversarialReportGenerator

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

    attacks: list[dict] = [{"id": atk} for atk in AF.get_list_classes(task=tasks)]
    requested_attacks = config.get("attacks", [])
    if requested_attacks:
        ids = AF.get_list_classes(task=tasks)
        attacks = [atk for atk in requested_attacks if atk.get("id", None) in ids]

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
        "Running benchmark: %d model(s), %d dataset(s), %d attack(s), %d metric(s)",
        len(models),
        len(datasets),
        len(attacks),
        len(metrics),
    )
    logger.info(
        "Output path: %s | Ray: %s | PDF report: %s",
        options.output_path, options.use_ray, options.create_pdf,
    )

    results: list[ModelReportProps] = run_benchmark(
        models=models,
        datasets=datasets,
        attacks=attacks,
        metrics=metrics,
        options=options,
    )

    #################################### PDF generation ####################################
    # Optionally, after the benchmark, it could be created the PDF report of the vulnerabilities

    if options.create_pdf:
        for result in results:
            report = AdversarialReportGenerator()
            report.generate(
                data=result,
                output_path=f"{options.output_path}/report.pdf",
                header_logo_path=None,
            )
    ###########################################################################################
