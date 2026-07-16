import os
from pathlib import Path


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
