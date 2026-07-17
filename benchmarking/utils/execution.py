import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from benchmarking.utils import evaluate_attack
from models import DatasetInfo, ModelInfo, BenchmarkOptionConfig
from nn_trust import ModelAdapter, StatisticsFactory as SF
from utils import load_model, get_dataloader


def execute_job(
        benchmark_id: str,
        dataset_cnf: DatasetInfo,
        model_cnf: ModelInfo,
        attack: dict,
        metrics: list[dict],
        options: BenchmarkOptionConfig,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> dict:
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

    ############################### 2) Loading the DataLoader and Model ###############################
    dataloader: DataLoader = get_dataloader(
        dataset_path=dataset_cnf.repository,
        batch=dataset_cnf.batch_size,
        subset=dataset_cnf.num_samples,
        transform=transformation,
        num_workers=dataset_cnf.num_workers,
        name=dataset_cnf.name
    )

    model: ModelAdapter = load_model(
        model_id=model_cnf.id or model_cnf.name,
        model_type=model_cnf.type,
        model_path=model_cnf.repository,
        api_url=model_cnf.api,
        task=model_cnf.task,
        device=device
    )

    # Add metric-specific defaults only when supported
    statistics = [dict(metric) for metric in metrics]

    targeted = getattr(options, "targeted", False)

    for metric in statistics:

        config_fields = SF.get_info(id=metric.get("id", None)).class_type.CONFIG_T.model_fields

        if "model" in config_fields:
            metric.setdefault("model", model)

        if "targeted" in config_fields:
            metric.setdefault("targeted", targeted)

        if "average_method" in config_fields:
            metric.setdefault("average_method", "macro")

    ##################################################################################################

    atk_config = {
        "name": attack.get("id", None),
        "id": attack.get("id", None),
        "targeted": targeted,
        **{
            key: value
            for key, value in attack.items()  # This is for extracting all the eventual parameters that are passed
            if key != "id"
        },
    }

    res = evaluate_attack(
        model=model,
        dataloader=dataloader,
        attack_config=atk_config,
        statistics=statistics,
        num_classes=dataset_cnf.num_classes,
        device=device,
        benchmark_id=benchmark_id,
    )
    return {
        "attack_results": res,
        "benchmark_job_info": benchmark_id,
    }
