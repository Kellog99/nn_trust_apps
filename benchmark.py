import argparse
import json
from pathlib import Path
from pprint import pprint

from benchmarking import run_benchmark
from models import BenchmarkOptionConfig, ModelInfo, DatasetInfo
from nn_trust import AttackFactory as AF, Task
from nn_trust.loss.loss_factory import LossFactory as LF

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Attack benchmark.')

    parser.add_argument(
        '--model_path',
        '-mp',
        required=True,
        help="Path to the model's information."
    )

    parser.add_argument(
        '--dataset_path',
        '-dp',
        required=True,
        help="Path to the dataset's information."
    )
    parser.add_argument(
        '--attacks',
        '-atk',
        default=[],
        help="List of attacks to run. By default all attacks are run."
    )
    parser.add_argument(
        '--metrics',
        '-m',
        default=[],
        help="List of metric to use. By default all metrics are used."
    )
    parser.add_argument(
        '--output_path',
        '-op',
        default="./tmp",
        help="Path where to store all the information from the benchmark."
    )
    parser.add_argument(
        '--use_ray',
        '-ray',
        action="store_true",
        default=False,
        help="Whether to use ray or not. Hence, whether to parallelize the attacks or not."
    )
    args = parser.parse_args()

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
