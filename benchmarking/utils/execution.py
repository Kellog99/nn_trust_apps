import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from benchmarking.utils import get_dataloader, evaluate_attack
from models import DatasetInfo, ModelInfo, BenchmarkOptionConfig
from nn_trust import ModelAdapter
from nn_trust.evaluation.statistic_factory import StatisticsFactory as SF
from utils.load_model import load_model


def override_keys_if_not_none(base_dict: dict, overriding_dict: dict) -> dict:
    """override keys in base dictionary using overriding dict if the latter are not none.
    Or the original dict do not contain the key to begin with"""
    res = dict(base_dict)  # create copy not to modify original element
    for k, v in overriding_dict.items():
        if v is not None or k not in res:
            res[k] = overriding_dict[k]
    return res


# The job_config is used so the executor can pass a complete inflated benchmark job, including model, dataset, attack, metrics, options, and benchmark metadata.
def execute_job(
        dataset_cnf: DatasetInfo,
        model_cnf: ModelInfo,
        attack: dict,
        metrics: list[dict],
        options: BenchmarkOptionConfig,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> None:
    """
    A function that use a full description of benchmark configuration and executor, is tasked to execute benchmark.
    """
    ############################### 1) defining the transformation ###############################
    transformation = [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=getattr(model_cnf.transformation, "mean", (0.5, 0.5, 0.5)),
            std=getattr(model_cnf.transformation, "std", (0.5, 0.5, 0.5))
        )
    ]
    if getattr(model_cnf.transformation, "size", None) is not None:
        transformation.append(transforms.Resize((model_cnf.transformation.size, model_cnf.transformation.size)))
    if getattr(model_cnf.transformation, "crop", None) is not None:
        transformation.append(transforms.CenterCrop(model_cnf.transformation.crop))
    transformation = transforms.Compose(transformation)
    ##############################################################################################

    ############################### 2) Loading the DataLoader and Mode ###############################
    dataloader: DataLoader = get_dataloader(
        dataset_path=dataset_cnf.repository,
        batch=dataset_cnf.batch_size,
        subset=dataset_cnf.num_samples,
        transform=transformation,
        num_workers=dataset_cnf.num_workers,
        name=dataset_cnf.name
    )

    model: ModelAdapter = load_model(
        model_path=model_cnf.repository,
    )
    model = model.to(device)
    model.eval()

    # Add metric-specific defaults only when supported
    statistics = [dict(metric) for metric in metrics]

    targeted = getattr(options, "targeted", False)

    for metric in statistics:
        metric_name = metric["name"]
        config_fields = SF.get_info(metric_name).class_type.CONFIG_T.model_fields

        if "model" in config_fields:
            metric.setdefault("model", model)

        if "targeted" in config_fields:
            metric.setdefault("targeted", targeted)

        if "average_method" in config_fields:
            metric.setdefault("average_method", "macro")

    ##################################################################################################

    atk_config = {
        "name": attack.id,
        "id": attack.id,
        "targeted": targeted,

        **{
            param.id: param.default
            for param in attack.parameters
            if param.default is not None
        },
    }

    res = evaluate_attack(
        model=model,
        dataloader=dataloader,
        attack_config=atk_config,
        statistics=statistics,
        num_classes=dataset_cnf.num_classes,
        device=device,
        benchmark_id=benchmark_info["benchmark_id"],
    )
    return {"attack_results": res,
            "benchmark_job_info": benchmark_info,
            }
