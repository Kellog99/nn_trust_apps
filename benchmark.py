import argparse
import json
from pathlib import Path

import yaml

from benchmarking import run_benchmark
from models import BenchmarkOptionConfig, ModelInfo, DatasetInfo
from nn_trust import AttackFactory as AF, Task
from nn_trust.loss.loss_factory import LossFactory as LF


def valid_dataset_info(data: dict) -> DatasetInfo:
    """
    Retrieve the information of the dataset
    """
    source_path = data.get('source_path', None)
    if source_path:
        source_path = Path(source_path).expanduser() / "info.json"
        if source_path.exists():
            with open(source_path, 'rb') as f:
                info_data = json.load(f)

            # Retrieve the information from the info json
            return DatasetInfo.model_validate(info_data)

    # Check whether there are information in the configuration file
    return DatasetInfo.model_validate(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Attack benchmark.')

    parser.add_argument(
        '--config_path',
        '-cnf',
        required=True,
        help="Path to the Benchmark configuration file."
    )

    args = parser.parse_args()
    config_path: Path = Path(args.config_path).expanduser()
    if not config_path.exists():
        raise ValueError(f"The path to the Benchmark configuration file does not exist: {config_path}")
    with open(args.config_path) as f:
        config = yaml.safe_load(f)

    import pdb

    pdb.set_trace()

    ####################### 1) Getting the models #######################
    model_path: Path = Path(args.model_path).expanduser()
    if not model_path.exists():
        raise FileNotFoundError("The path to the model's location does not exist.")
    with open(model_path, 'rb') as f:
        data = json.load(f)
    model_info: ModelInfo = ModelInfo.model_validate(data)
    if getattr(model_info, 'repository', None) is None:
        model_info.repository = str(model_path.parent)
    #####################################################################

    ####################### 2) Getting the dataset #######################
    dataset_path: Path = Path(args.dataset_path).expanduser()
    if not dataset_path.exists():
        raise FileNotFoundError("The path to the dataset's location does not exist.")
    with open(dataset_path, 'rb') as f:
        data = json.load(f)
    dataset: DatasetInfo = DatasetInfo.model_validate(data)
    if getattr(dataset, 'repository', None) is None:
        dataset.repository = str(dataset_path.parent)
    ######################################################################

    ####################### 3) Getting the attacks #######################
    list_atk = AF.get_list_classes(task={Task.from_str(model_info.task)})
    if len(args.attacks) > 0:
        list_atk = [atk for atk in list_atk if atk in args.attacks]
    ######################################################################

    ####################### 4) Getting the metrics #######################
    list_metrics = LF.get_list_classes(task={Task.from_str(model_info.task)})
    if len(args.metrics) > 0:
        list_metrics = [mtr for mtr in list_metrics if mtr in args.metrics]
    ######################################################################

    ####################### Setting the Benchmarking options #######################
    options: BenchmarkOptionConfig = BenchmarkOptionConfig(
        num_images_to_save=10,
        save_perturbation=True,
        output_path=args.output_path,
        use_ray=args.use_ray,
    )
    print("Executing benchmark with following configuration:\n")
    run_benchmark(
        models=[model_info],
        datasets=[dataset],
        attacks=args.attacks,
        metrics=list_metrics,
        options=options,
    )
    print("\nDone")
