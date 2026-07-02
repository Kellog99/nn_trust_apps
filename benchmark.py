from benchmarking import config_file_path_selector, run_benchmark

if __name__ == "__main__":
    selected_config_path = config_file_path_selector(Path(__file__).parent / "config")
    print("Executing benchmark with following configuration:\n")
    run_benchmark({})
    print("\nDone")
