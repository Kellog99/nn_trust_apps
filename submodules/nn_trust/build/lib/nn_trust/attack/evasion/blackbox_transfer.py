from typing import Optional, Union

import torch
import torch.nn as nn
import torchvision.models
from pydantic import Field, field_validator, model_validator
from torch.utils.data import ConcatDataset, DataLoader
from tqdm.auto import tqdm

from nn_trust.attack import EvasionAttackFactory
from nn_trust.attack._evasion import EvasionAttack, EvasionAttackConfig
from nn_trust.attack.utils._utils import to_device
from nn_trust.core import AttackType, Knowledge, ModelAdapter, Task


class BlackBoxTransferAttackConfig(EvasionAttackConfig):
    dataset_augmentation: float = Field(
        default=0.1,
        description="The maximum norm the dataset original images should be augmented with.",
        gt=0.0,
    )
    optimizer_type: type[torch.optim.Optimizer] = Field(
        default=torch.optim.Adam,
        description="The type of Optimizer to use."
    )
    optimizer_args: dict[str, int | float] = Field(
        default_factory=lambda: dict(lr=1e-3, weight_decay=1e-3),
        description="Arguments to pass to the optimizer."
    )

    surrogate_model: ModelAdapter = Field(
        default="cifar10dummy",
        description="Surrogate model to use for the attack.",
        validate_default=True,
    )

    attack: Optional[EvasionAttack] = Field(
        default=None,
        description="Evasion attack algorithm to reach classification boundary."
    )

    @field_validator("surrogate_model", mode="before")
    @classmethod
    def validate_surrogate_model(cls, val: Union[str, ModelAdapter]):
        # Create a few backbones
        if isinstance(val, str):
            match val:
                case "ResNet34":
                    val = ModelAdapter(model=torchvision.models.resnet34(pretrained=True))
                case "ResNet18":
                    val = ModelAdapter(model=torchvision.models.resnet18(pretrained=True))
                case "vgg16":
                    val = ModelAdapter(model=torchvision.models.vgg16(pretrained=True))
                case "cifar10dummy":
                    cifar10_dummy = torchvision.models.resnet18(weights=None)
                    cifar10_dummy.fc = torch.nn.Sequential(torch.nn.Flatten(1), torch.nn.LazyLinear(10))
                    val = ModelAdapter(model=cifar10_dummy)

        return val

    @model_validator(mode="after")
    def validate_sub_attack(self):
        if self.attack is None:
            self.attack = EvasionAttackFactory.create(
                class_id="fgsm",
                model=self.model,
                task=self.task
            )
        return self


@EvasionAttackFactory.register(
    name="Black Box Transfer",
    task={Task.Classification},
    description="Use a surrogate model to distill a black box model's knowledge and attack the former in a white-box manner.",
    type=AttackType.Digital,
    knowledge=Knowledge.Black
)
class BlackBoxTransferAttack(EvasionAttack):
    r"""Implements the transferability attack by exploiting a black box model to train a ``surrogate_model`` model
     that should resemble it[1]_. Finally, we harness the surrogate model potential similarity to a target model
     to generate potential adversarial attack to the original model. The goal is to reduce the number of queries
     with respect to the Black Box model, in order for the attack to be less detectable from APIs call in case
     of a inference server.

    .. [1] Papernot, Nicolas, Patrick Mcdaniel, Ian J. Goodfellow, Somesh Jha, Z. Berkay Celik and Ananthram Swami. “Practical Black-Box Attacks against Machine Learning.” Proceedings of the 2017 ACM on Asia Conference on Computer and Communications Security (2016): n. pag.
    """

    CONFIG_T = BlackBoxTransferAttackConfig

    def __init__(self, config: BlackBoxTransferAttackConfig):
        super().__init__(config)
        self.config.surrogate_model = self.config.validate_surrogate_model(self.config.surrogate_model)

    def fit(self, proxy_dataloader: torch.utils.data.DataLoader, **kwargs) -> None:
        """Trains a surrogate model based on the proxy dataset and a against a known model, i.e.
        the model to attack."""
        # Train the backbone
        to_device(self.config, self.config.device)
        self.config.surrogate_model.train(True)
        optimizer = self.config.optimizer_type(
            self.config.surrogate_model.parameters(), **self.config.optimizer_args
        )
        loss = nn.CrossEntropyLoss()
        loop = range(1, self.config.max_iters + 1)
        if self.config.verbose:
            loop = tqdm(loop)

        for _ in loop:
            new_dataset_chunk = []
            training_default_device = "cpu"
            avg_loss = 0.0
            for _i, (x, _) in enumerate(proxy_dataloader):
                if x.device != training_default_device:
                    training_default_device = x.device

                optimizer.zero_grad()
                x = x.to(self.config.device)
                if x.dim() == 3:
                    x = x.unsqueeze(0)
                # Compute the new images using the model
                x.requires_grad_(True)
                oracle_inference = self.config.model(x)
                oracle_inference_argmax = oracle_inference.argmax(dim=-1).detach()
                x_loss = loss(oracle_inference, oracle_inference_argmax)
                x_loss.backward()
                x_prime = (x + self.config.dataset_augmentation * x_loss.sign()).detach()
                x_prime.requires_grad_(False)
                # Fine tune the surrogate model
                x = x.detach()
                x.requires_grad_(False)
                backbone_loss_val = loss(self.config.surrogate_model(x), oracle_inference_argmax)
                backbone_loss_val.backward()
                optimizer.step()
                avg_loss += backbone_loss_val.item()
                # dataset chunks definition
                oracle_inference_argmax = oracle_inference_argmax.to(training_default_device)
                x_prime = x_prime.to(training_default_device)
                one_hot_inference = torch.nn.functional.one_hot(
                    oracle_inference_argmax, num_classes=oracle_inference.shape[-1]
                )
                new_dataset_chunk = list(zip(x_prime.unbind(0), one_hot_inference.unbind(0), strict=False))

            if self.config.verbose:
                loop.set_postfix({"backbone loss": avg_loss / (_i + 1)})

            # Augment the dataset using the model to attack's gradient
            if len(new_dataset_chunk) > 0:
                # check one-hot encoding
                if new_dataset_chunk[0][1].size(-1) != proxy_dataloader.dataset[0][1].size(-1):
                    raise ValueError(
                        f"The proxy dataloader does not load its labels as one-hot encoded: {new_dataset_chunk[0][1].size(-1)} != {proxy_dataloader.dataset[0][1].size(-1)}"
                    )
                proxy_dataloader = DataLoader(
                    ConcatDataset([proxy_dataloader.dataset, new_dataset_chunk]),
                    shuffle=True,
                    batch_size=proxy_dataloader.batch_size,
                )

        self.config.surrogate_model.eval()

    def reset_fit(self):
        self.config.surrogate_model = self.config.validate_surrogate_model(self.config.backbone)

    def generate(
            self,
            x: torch.Tensor,
            y: Optional[torch.Tensor] = None,
            ext_results: Optional[dict] = None,
            **kwargs,
    ) -> torch.Tensor:
        to_device(self.config, self.config.device)
        if len(y.size()) == 1:
            y = y.repeat(x.size(0), 1)
        result = self.config.attack.generate(x, y, ext_results, **kwargs)
        return result