
import torch
from torchvision.models import resnet50, ResNet50_Weights

from nn_trust.attack import EvasionAttackFactory, EvasionAttackConfig
from nn_trust.core import ModelAdapter, Task, Knowledge
import logging


def run_attack(
        img: torch.Tensor,
        attack_name:str,
        epsilon:float,
        p:float,
        max_iters:int
        
    ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img = img.unsqueeze(0).to(device)
    logging.info(f"Running attack on {device}")
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
        p=p,
        max_iters=max_iters
    )

    atk = EvasionAttackFactory.create_attack(attack_type=attack_name, config=cnf)
    labels = model_ad(img).argmax(1)
    labels_ohe = torch.nn.functional.one_hot(labels, num_classes=1000)
    print("****************")
    print(labels_ohe.shape)

    x_adv = atk.generate(x=img, y=labels_ohe)
    labels_adv = model_ad(x_adv).argmax(1)
    return x_adv[0], labels.item(), labels_adv.item()
