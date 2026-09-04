import argparse
import logging
from pprint import pprint
from pathlib import Path

import yaml

from benchmarking import run_benchmark, load_info
from models import BenchmarkOptionConfig, ModelInfo, DatasetInfo, ModelReportProps
from nn_trust import AttackFactory as AF, Task
from report import AdversarialReportGenerator

logger = logging.getLogger("benchmark")


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

    ####################### 1) Getting the model #######################
    models = [load_info(entry, ModelInfo) for entry in config.get("model", [])]
    if not models:
        raise ValueError("The configuration must define at least one entry under 'model'.")
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

    ####################### Setting the Benchmarking options #######################
    options = BenchmarkOptionConfig.model_validate(config.get("options", {}))

    logger.info(
        "Running benchmark: %d model(s), %d dataset(s), %d attack(s), %d metric(s)",
        len(models),
        len(datasets),
        len(attacks),
        len(config.get("metrics", [])),
    )
    logger.info(
        "Output path: %s | Ray: %s | PDF report: %s",
        options.output_path, options.use_ray, options.create_pdf,
    )

    results: list[ModelReportProps] = run_benchmark(
        models=models,
        datasets=datasets,
        attacks=attacks,
        metrics=config.get("metrics", []),
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
