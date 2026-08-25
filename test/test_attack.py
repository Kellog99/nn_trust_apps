import pytest
import torch
import torchvision
from PIL.Image import Image

from nn_trust import CVModelAdapter, Task, AttackObjective
from nn_trust.attack import AttackFactory as AF, EvasionAttack
from services.utils.attack import single_attack_performance
from test.utils.utils import get_dog_image, get_dummy_cv_model


def available_devices():
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    return devices


@pytest.mark.parametrize("model", [get_dummy_cv_model()])
@pytest.mark.parametrize(
    "attack_id",
    AF.get_list_classes(
        task={Task.Classification},
        objective=[AttackObjective.EVASION],
    )[:2],
)
@pytest.mark.parametrize("pil_image", [get_dog_image()])
@pytest.mark.parametrize("device", available_devices()[1:3])
def test_attack(
        model: CVModelAdapter,
        attack_id: str,
        pil_image: Image,
        device: torch.device
):
    """
    Verify that every evasion attack runs without errors.
    """

    print(sum(
        p.numel() for p in model.parameters()
        if not isinstance(p, torch.nn.parameter.UninitializedParameter)
    ))

    cnf = AF.get_config(
        class_id=attack_id,
        max_iters=3,
        model=model.to(device),
        verbose=False,
        device=device,
    )
    if hasattr(cnf, "surrogate_model"):
        sm = torchvision.models.resnet18()
        sm.fc = torch.nn.Sequential(torch.nn.Flatten(1), torch.nn.LazyLinear(10))
        sm.eval()
        setattr(cnf, "surrogate_model", sm.to(device))

    atk: EvasionAttack = AF.create(config=cnf)
    out = single_attack_performance(
        model=model,
        attack=atk,
        pil_image=pil_image,
        device=device
    )
