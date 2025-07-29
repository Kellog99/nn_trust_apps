
import torch
from torchvision.models import resnet50, ResNet50_Weights

from nn_trust.attack import EvasionAttackFactory, EvasionAttackConfig
from nn_trust.core import ModelAdapter, Task, Knowledge
import logging


def run_attack(
        img: torch.Tensor,
        attack_name:str,
        epsilon:float,
        p:float
        
    ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.INFO(f"Running attack on {device}")
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.eval()
    model.to(device)
    model_ad = ModelAdapter(model, Knowledge.Black)

    cnf = EvasionAttackFactory.get_config(
        attack_type=attack_name,
        model=model_ad,
        task=Task.Classification,
        device=device,
        verbose=True,
        epsilon=epsilon,
        p=p
    )

    atk = EvasionAttackFactory.create_attack(attack_type=attack_name, config=cnf)

    images = torch.rand((10,3, 224, 224)) # BxCxHxW
    labels_ohe = torch.zeros(10) # example with K=10 classes
    labels_ohe[3] = 1 # example belong to class 4

    x_adv = atk.generate(x=images, y=labels_ohe)
    return x_adv
