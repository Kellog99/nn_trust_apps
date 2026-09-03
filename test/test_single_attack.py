import warnings

import pytest
import torch
import torchvision
from PIL.Image import Image

from models import SingleAttackOutput
from nn_trust import CVModelAdapter, AttackConfig
from nn_trust.attack import AttackFactory as AF, EvasionAttack
from services.utils.attack import single_attack_performance
from services.utils.utils import tensor_image_to_b64str
from test.utils import get_dummy_cv_model, available_devices, get_dog_image

"""@pytest.mark.parametrize(
    "attack_id",
    AF.get_list_classes(
        task={Task.Classification},
        objective=[AttackObjective.EVASION],
    )[:2],
)"""


@pytest.mark.parametrize(
    "attack_id",
    ["fom"],
)
@pytest.mark.parametrize("model", [get_dummy_cv_model()])
@pytest.mark.parametrize("pil_image", [get_dog_image()])
@pytest.mark.parametrize("device", available_devices())
def test_attack(
        model: CVModelAdapter,
        attack_id: str,
        pil_image: Image,
        device: torch.device
):
    """
    Verify that every evasion attack runs without errors.
    """
    params: int = sum(
        p.numel() for p in model.parameters()
        if not isinstance(p, torch.nn.parameter.UninitializedParameter)
    )
    print(f"params: {params}")

    cnf: AttackConfig = AF.get_config(
        class_id=attack_id,
        max_iters=3,
        model=model.to(device),
        verbose=False,
        device=device,
        early_stopping=False
    )
    if hasattr(cnf, "surrogate_model"):
        sm = torchvision.models.resnet18()
        sm.fc = torch.nn.Sequential(torch.nn.Flatten(1), torch.nn.LazyLinear(10))
        sm.eval()
        setattr(cnf, "surrogate_model", sm.to(device))

    atk: EvasionAttack = AF.create(config=cnf)
    out: SingleAttackOutput = single_attack_performance(
        model=model.to(device),
        attack=atk,
        pil_image=pil_image,
        device=device
    )
    for k in out.confidence.keys():
        assert len(out.confidence[k]) == cnf.max_iters


def test_tensor_image_to_b64str_sanitizes_attack_output():
    """Image serialization must not warn for invalid attack tensor values."""
    image = torch.tensor([[[float("nan"), float("inf")], [-float("inf"), 2.0]]])

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        encoded = tensor_image_to_b64str(image)

    assert encoded
    assert not [warning for warning in record if issubclass(warning.category, RuntimeWarning)]


if __name__ == "__main__":
    pil_image: Image = get_dog_image(num_images=1)
    test_attack(
        model=get_dummy_cv_model(),
        pil_image=pil_image,
        device=torch.device("cpu"),
        attack_id="fom"
    )
