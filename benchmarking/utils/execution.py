import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from benchmarking.utils import get_dataloader, evaluate_attack
from models import DatasetInfo, ModelInfo, RegisteredObject
from nn_trust import ModelAdapter, AttackFactory
from nn_trust.models.model_utils import load_model


def override_keys_if_not_none(base_dict: dict, overriding_dict: dict) -> dict:
    """override keys in base dictionary using overriding dict if the latter are not none.
    Or the original dict do not contain the key to begin with"""
    res = dict(base_dict)  # create copy not to modify original element
    for k, v in overriding_dict.items():
        if v is not None or k not in res:
            res[k] = overriding_dict[k]
    return res


def execute_job(
        attack: RegisteredObject,
        metrics: list[RegisteredObject],
        model_cnf: ModelInfo,
        dataset_cnf: DatasetInfo,
):
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
    if hasattr(model_cnf.transformation, "size"):
        transformation.append(transforms.Resize((model_cnf.transformation.size, model_cnf.transformation.size)))
    if hasattr(model_cnf.transformation, "crop"):
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
    model: ModelAdapter = load_model(model_path=model_cnf.repository)
    ##################################################################################################

    atk_config = AttackFactory.get_config(
        class_id=attack.id,
        **{param.id: param.default for param in attack.parameters}
    )
    res = evaluate_attack(
        model=model,
        dataloader=dataloader,
        attack_config=atk_config,
        statistics=metrics,
        num_classes=dataset_cnf.num_classes,
    )
    return {"attack_results": res}
