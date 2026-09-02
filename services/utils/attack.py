import datetime
from pathlib import Path

import time
import torch
from PIL import Image
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchvision.transforms import v2 as T, InterpolationMode

from models import SingleAttackOutput
from nn_trust import CVModelAdapter
from nn_trust.attack import EvasionAttack
from nn_trust.target import AvoidOnehotTarget
from nn_trust.utils.logger import PyTorchCheckpointLogger
from services.utils.utils import tensor_image_to_b64str


def single_attack_performance(
        model: CVModelAdapter,
        attack: EvasionAttack,
        pil_image: Image.Image,
        input_dimensionality: list[int] | int = [224, 224],
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
) -> SingleAttackOutput:
    """
    Perform a single attack with a
    """
    ################## IMAGE ##################
    if isinstance(input_dimensionality, list):
        if len(input_dimensionality) == 3:
            input_dimensionality = input_dimensionality[1:]
        elif len(input_dimensionality) == 1:
            input_dimensionality = [input_dimensionality[0], input_dimensionality[0]]
        input_dimensionality = tuple(input_dimensionality)
    elif isinstance(input_dimensionality, int):
        input_dimensionality = (input_dimensionality, input_dimensionality)

    ############ image transformation ############
    original_input: torch.Tensor = T.ToTensor()(pil_image)
    C, H, W = original_input.shape

    transformations = T.Compose([
        T.Resize(size=input_dimensionality),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    x: torch.Tensor = transformations(pil_image)
    if x.dim() == 3:
        x = x.unsqueeze(0)
    x: torch.Tensor = x.to(device)
    print(" Image loaded ".center(40, "#"))

    ###############################################

    ################## Results ##################
    y = model(x)
    labels = y.argmax(-1).tolist()
    target = AvoidOnehotTarget(num_classes=y.shape[-1])(labels).to(device)

    # Logger for getting additional material
    out_path = Path(f"./tmp/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    logger = PyTorchCheckpointLogger(
        states=["conf_adversarial", "conf_original"],
        path=out_path
    )
    print(" Executing the Attack ".center(40, "#"))

    start = time.time()
    x_adv = attack.generate(
        x=x,
        y=target,
        logger=logger
    ).detach()
    end = time.time()

    with torch.no_grad():
        print(" Attack Completed ".center(40, "#"))
        y_adv: torch.Tensor = model(x_adv).argmax(-1)
        ssim_metric = StructuralSimilarityIndexMeasure().to(device)
        ssim_measure: float | int = ssim_metric(x.to(device), x_adv.to(device)).item()
        print(ssim_measure)

    conf_original: dict = logger.get_log(tag="conf_original", state="generate")
    conf_adversarial: dict = logger.get_log(tag="conf_adversarial", state="generate")
    ############################################

    ################## Invert transform ################
    inv_transform = T.Compose([
        T.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
        ),
        T.Resize(size=(H, W), interpolation=InterpolationMode.BICUBIC),
    ])
    print(x.shape)
    pert: torch.Tensor = x_adv.cpu() - x.cpu()
    print("pert = ", pert.shape)
    pert = inv_transform(pert)
    print("pert = ", pert.shape)
    x_adv: torch.Tensor = inv_transform(x_adv.cpu())

    return SingleAttackOutput(
        x_adv=tensor_image_to_b64str(x_adv.cpu()),
        adv_perturbation=tensor_image_to_b64str(pert.cpu()),
        original_prediction=str(labels[0]),
        adversarial_prediction=str(y_adv.item()),
        advance_metrics={
            "ssim": ssim_measure,
            "distance": torch.norm(pert, p=getattr(attack.config, "p", 2)).item(),
            "execution_time": end - start,
        },
        confidence={
            "adversarial": conf_original,
            "original": conf_adversarial,
        }
    )
