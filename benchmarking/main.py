import json
from pathlib import Path
import os
import logging
import traceback
from datetime import datetime
import pathlib

from nn_trust.attack.evaluation.composer import ConfigStatisticComposer, StatisticComposer

try:
    from benchmark_utils import (
        read_config_file,
        BenchmarkConfig,
        Evaluator,
        get_structure,
        config_file_path_selector
    )
    from benchmark_utils.pdf_report import create_benchmark_report
except ModuleNotFoundError:
    # when used from attack.server it needs to import as if working as a module
    from .benchmark_utils import (
        read_config_file,
        BenchmarkConfig,
        Evaluator,
        get_structure,
        config_file_path_selector
    )
    from .benchmark_utils.pdf_report import create_benchmark_report

def benchmark_(config: dict):
    output_path = Path(config["options"]["output_path"])
    output_path = output_path / datetime.now().strftime("%Y%m%dT%H%M%S")
    os.makedirs(output_path, exist_ok=False)
    config["output_path"] = str(output_path)
    logging.info(f"Benchmark run will be save to {output_path}")
    for dataset_id, dataset in enumerate(config["datasets"]):
        # for i, model_id in enumerate(config["model"]["list_models"]):
        for model_id, model_config in enumerate(config["models"]):

            # transformations should depend on dataset and model
            try:
                evaluator = Evaluator.from_config(config=config, dataset=dataset, model_config=model_config)
                evaluator.evaluate_attacks()
                evaluator.save_results_to_disk()
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

def benchmark(config: dict):
    output_path = benchmark_(config)
    postprocess_benchmark_run_results(output_path)


def main():
    handler = logging.StreamHandler()
    handler.addFilter(lambda record: record.name == "root")
    logging.basicConfig(level=logging.WARN, handlers=[handler])
    # get the parser
    selected_config_path = config_file_path_selector(Path(__file__).parent / "config")
    config = read_config_file(config_filename=str(selected_config_path))
    config = BenchmarkConfig(**config)

    output_path = benchmark_(config.model_dump())
    print(f"Results saved to {output_path}")
    postprocess_benchmark_run_results(output_path)

    models = config.model_dump()["models"]
    datasets = config.model_dump()["datasets"]
    for dataset in datasets:
        dataset_name = dataset["name"]
        for model in models:
            model_name = model["name"]
            print(f"Generating report for {dataset_name} - {model_name}")
            create_benchmark_report(
                benchmark_run_dir=output_path,
                model_name=model_name,
                dataset_name=dataset_name,
                filename=Path(output_path) / dataset_name / model_name / "report.pdf",
                generated_by="Leonardo S.p.A."
            )

if __name__ == "__main__":
    main()





