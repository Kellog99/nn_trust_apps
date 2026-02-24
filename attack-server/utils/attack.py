import time

import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure

from models.attack import SingleAttackOutput
from nn_trust import ModelAdapter
from nn_trust.attack import EvasionAttack
from nn_trust.target import AvoidOnehotTarget


def single_image_attack(
        model: ModelAdapter,
        x: torch.Tensor,
        attack: EvasionAttack,
        device: torch.device = torch.device("cpu"),
) -> SingleAttackOutput:
    """
    This function handle the execution of one image attack and generating all the metrics that are required.
    """
    model = model.to(device)
    model.eval()

    if x.dim() == 3:
        x = x.unsqueeze(0)
    x = x.to(device)

    y = model(x)
    labels = y.argmax(-1).tolist()
    target = AvoidOnehotTarget(num_classes=y.shape[-1])(labels).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure()

    start = time.time()
    x_adv = attack.generate(x=x, y=target)
    end = time.time()

    pert = x_adv.cpu() - x
    y_adv = model(x_adv).argmax(-1)
    ssim_measure = ssim_metric(x, x_adv.cpu()).item()

    return SingleAttackOutput(
        x_adv=x_adv.cpu(),
        adv_perturbation=pert.cpu(),
        original_prediction=str(labels[0]),
        adversarial_prediction=str(y_adv.item()),
        advance_metrics={
            "ssim": ssim_measure,
            "distance": torch.norm(pert, p=getattr(attack.config, "p", 2)).item(),
            "execution_time": end - start,
        }
    )
