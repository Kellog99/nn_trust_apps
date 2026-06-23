from pathlib import Path

from benchmarking import config_file_path_selector, read_config_file
from benchmarking.benchmark_utils import run_benchmark_with_configuration

if __name__ == "__main__":
    selected_config_path = config_file_path_selector(Path(__file__).parent / "config")
    config = read_config_file(config_filename=str(selected_config_path))
    print("Excuting benchmark with following configuration:\n")
    print("-" * 30)
    print(config)
    print("-" * 30)
    run_benchmark_with_configuration(config)
    print("\nDone")
