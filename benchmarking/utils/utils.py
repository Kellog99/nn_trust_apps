import json
import os
import traceback
from pathlib import Path

from benchmarking.utils.evaluation import read_results_from_disk, aggregate_attacks_statistics
from nn_trust.evaluation.composer import StatisticComposer, ConfigStatisticComposer


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


def postprocess_results(
        benchmark_run_dir: str | Path,
        verbose=True
):
    """
    Walk each (dataset, model) subdirectory of a completed benchmark run,
    compute aggregate statistics across its attacks, and write them back to
    disk alongside a re-serialized info.json.

    A failure on one subdirectory is logged and skipped rather than aborting
    the whole postprocessing pass.
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
            traceback.print_exc()
