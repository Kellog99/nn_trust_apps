from pathlib import Path

try:
    from .benchmark_utils import (
        read_config_file, 
        config_file_path_selector, 
        run_benchmark_with_configuration
    )
except:
    from benchmark_utils import (
        read_config_file, 
        config_file_path_selector, 
        run_benchmark_with_configuration
    ) 

def main():
    selected_config_path = config_file_path_selector(Path(__file__).parent / "config")
    config = read_config_file(config_filename=str(selected_config_path))
    print("Excuting benchmark with following configuration:\n")
    print("-"*30)
    print(config)
    print("-"*30)
    run_benchmark_with_configuration(config)
    print("\nDone")

if __name__ == "__main__":
    main()




