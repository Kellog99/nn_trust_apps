import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from benchmarking.utils import evaluate_attack
from models import DatasetInfo, ModelInfo, BenchmarkOptionConfig
from models.benchmark import AttackEvaluation
from nn_trust import ModelAdapter, StatisticComposer, StatisticsFactory as SF, LossComposer, AttackFactory as EAF, Task
from nn_trust.attack import EvasionAttack
from utils import load_model, get_dataloader


def execute_job(
        benchmark_id: str,
        dataset_cnf: DatasetInfo,
        model_cnf: ModelInfo,
        attack_cnf: dict,
        metrics: list[dict],
        options: BenchmarkOptionConfig,
) -> AttackEvaluation:
    """
    A function that use a full description of benchmark configuration and executor, is tasked to execute benchmark.
    Args
        :param benchmark_id: benchmark id
        :param dataset_cnf: dataset info
        :param model_cnf: model info
        :param metrics: this represents the list with all the parameters for instantiate the statistic composer
        :param attack_cnf: list of all the attacks to execute with all the parameters for instantiate them
        :param options: represent all the possible options that could be used during the benchmark
    """
    # 0) Setting the device
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() and options.gpu else "cpu")
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
        subset=options.subset,
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
    print(len(dataloader.dataset))
    #################### Defining the Statistic Composer ####################
    batch, _ = next(iter(dataloader))
    out: torch.Tensor = model(batch.to(device))
    num_classes: int = out.shape[-1]

    metrics: list[dict] = [
        {
            **metric,
            "model": model,
            "device": device,
            "num_classes": num_classes,
        }
        for metric in metrics if metric.get("id") in SF.get_list_classes()
    ]
    statistics_composer = StatisticComposer(
        statistics=metrics,
        device=device
    )
    #################### Defining the Statistic Composer ####################

    ########################## Defining the Attack ##########################
    atk_id: str = attack_cnf.get("id", None) or attack_cnf.get("name", None)
    if atk_id is None:
        raise ValueError("No id for instantiate the attack")

    atk_config = {
        "name": attack_cnf.get("id", None),
        "id": attack_cnf.get("id", None),
        **{
            key: value
            for key, value in attack_cnf.items()  # This is for extracting all the eventual parameters that are passed
            if key != "id"
        },
    }
    # Checking whether some losses have to be set
    if atk_config.get("losses", None) is not None:
        # If losses are specified, convert them to Loss objects
        atk_config['loss'] = LossComposer(
            losses=atk_config['losses'],
            weights=atk_config.get('loss_weights', [1.0] * len(atk_config['losses'])),
        )

    atk: EvasionAttack = EAF.create(
        class_id=atk_id,
        model=model,
        device=device,
        task=Task.Classification,
        **atk_config
    )
    #################### Defining the Statistic Composer ####################

    res: AttackEvaluation = evaluate_attack(
        model=model,
        dataloader=dataloader,
        attack=atk,
        statistics=statistics_composer,
        num_classes=num_classes,
        verbose=options.verbose,
        device=device
    )

    res.id = benchmark_id
    return res
