import datetime
from pathlib import Path

import time
import torch
from PIL import Image
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchvision.transforms import v2 as T, InterpolationMode

from models import SingleAttackOutput
from nn_trust import CVModelAdapter, EvasionAttack
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
    ################## IMAGE ##################
    if isinstance(input_dimensionality, list):
        if len(input_dimensionality) == 3:
            input_dimensionality = input_dimensionality[1:]
        elif len(input_dimensionality) == 1:
            input_dimensionality = [input_dimensionality[0], input_dimensionality[0]]
        input_dimensionality = tuple(input_dimensionality)
    elif isinstance(input_dimensionality, int):
        input_dimensionality = (input_dimensionality, input_dimensionality)

    # Force a fixed, model-expected channel count (fixes silent C-mismatch -> CUDA assert)
    pil_image = pil_image.convert("RGB")

    ############ image transformation ############
    original_input: torch.Tensor = T.ToTensor()(pil_image)
    print(original_input.shape)
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
    x = x.to(device)
    if not torch.isfinite(x).all():
        raise ValueError("Image preprocessing produced non-finite values.")
    print(" Image loaded ".center(40, "#"))
    ###############################################

    ################## Results ##################
    with torch.no_grad():
        y = model(x)
    if not torch.isfinite(y).all():
        raise RuntimeError(
            "The model produced non-finite logits for the original image. "
            "Check the model weights and preprocessing configuration."
        )
    labels = y.argmax(-1).tolist()
    num_classes: int = y.shape[-1]

    target = AvoidOnehotTarget(num_classes=num_classes)(labels).to(device)

    out_path = Path(f"./tmp/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    logger: PyTorchCheckpointLogger = PyTorchCheckpointLogger(
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
    if not torch.isfinite(x_adv).all():
        raise RuntimeError(
            "The attack produced non-finite values. Reduce the attack epsilon/learning rate "
            "or enable epsilon-ball projection."
        )

    with torch.no_grad():
        print(" Attack Completed ".center(40, "#"))
        adv_logits = model(x_adv)
        if not torch.isfinite(adv_logits).all():
            raise RuntimeError(
                "The model produced non-finite logits for the adversarial image. "
                "Reduce the attack epsilon/learning rate or enable epsilon-ball projection."
            )
        y_adv: torch.Tensor = adv_logits.argmax(-1)
        ssim_metric = StructuralSimilarityIndexMeasure().to(device)
        ssim_measure: float = ssim_metric(x.to(device), x_adv.to(device)).item()

    conf_original: list[list[float]] = logger.get_log(tag="conf_original")
    conf_adversarial: list[list[float]] = logger.get_log(tag="conf_adversarial")

    conf_original: list[float] = [conf[0] for conf in conf_original]
    conf_adversarial: list[float] = [conf[0] for conf in conf_adversarial]

    ############################################

    ################## Invert transform ################
    inv_transform = T.Compose([
        T.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
        ),
        T.Resize(size=(H, W), interpolation=InterpolationMode.BICUBIC),
    ])
    pert: torch.Tensor = x_adv.cpu() - x.cpu()
    # A perturbation has no mean component: inverse-normalize it by scaling
    # with the channel standard deviation only. Applying Normalize here would
    # add the ImageNet mean and inflate the reported distance.
    std = torch.tensor([0.229, 0.224, 0.225], dtype=pert.dtype).view(1, 3, 1, 1)
    pert = T.Resize(size=(H, W), interpolation=InterpolationMode.BICUBIC)(pert * std)
    x_adv = inv_transform(x_adv.cpu())

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
            "original": {i: conf for i, conf in enumerate(conf_original)},
            "adversarial": {i: conf for i, conf in enumerate(conf_adversarial)},
        }
    )
