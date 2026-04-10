from typing import Any, Optional

import torch
import torch.optim as optim
from pydantic import Field, model_validator

from nn_trust.attack import EvasionAttackFactory
from nn_trust.attack._evasion import EvasionAttack, EvasionAttackConfig
from nn_trust.attack.normalization import LpNormalization
from nn_trust.attack.utils._utils import _compare_misclassification, _compare_semantic
from nn_trust.core import AttackType, Knowledge, Task


class OFUAPAttackConfig(EvasionAttackConfig):
    delta: float = Field(
        default=0.01,
        description="Target error rate in classification.",
        ge=0.0,
        le=1.0
    )
    lp_norm: LpNormalization = Field(
        default_factory=lambda: LpNormalization(p=2.0, radius=8.0),
        description="Lp normalization for the perturbation."
    )
    attack: Optional[EvasionAttack] = Field(
        default=None,
        description="Evasion attack algorithm to reach classification boundary."
    )
    optimizer_type: type[optim.Optimizer] = Field(
        default=optim.SGD,
        description="The optimizer class to use during the attack."
    )
    optimizer_args: dict[str, Any] = Field(
        default_factory=lambda: dict(lr=0.01),
        description="The parameters to use in the optimizer."
    )

    @model_validator(mode="after")
    def validate_sub_attack(self):
        if self.attack is None:
            self.attack = EvasionAttackFactory.create(
                class_id="deepfool",
                model=self.model,
                task=self.task
            )
        return self


@EvasionAttackFactory.register(
    name="OLD Fast Universal Adversarial Perturbation Attack (FUAP)",
    description="A white-box universal adversarial attack utilizing a faster loss computation than UAP with minimal drawbacks in efficancy.",
    task={Task.Classification},
    type=AttackType.Digital,
    knowledge=Knowledge.White
)
class OFUAPAttack(EvasionAttack):
    CONFIG_T = OFUAPAttackConfig

    def __init__(self, config: EvasionAttackConfig):
        super().__init__(config)
        if self.config.model and self.config.attack.config.model is None:
            self.config.attack.config.model = self.config.model

    def track_variables(self):
        super().track_variables()
        self.add_variable_to_track("perturbation", "image")
        self.add_variable_to_track("fooling_ratio", "tensor")

    def _generate_aligned_perturbation(
            self,
            x: torch.Tensor,
            y: Optional[torch.Tensor] = None,
            init_perturbation: Optional[torch.Tensor] = None,
            **kwargs,
    ) -> torch.Tensor:
        # Starting perturbation is 0
        if init_perturbation is not None:
            perturbation = init_perturbation.to(self.config.device).requires_grad_(True)
        else:
            perturbation = torch.zeros_like(x, device=self.config.device, requires_grad=True)
        optimizer = self.config.optimizer_type([perturbation], **self.config.optimizer_args)

        loop = range(self.config.max_iters)
        # if self.config.verbose:
        #    loop = tqdm(loop, desc='Generating perturbation')
        for _ in loop:
            optimizer.zero_grad()
            adv_x = x + perturbation
            if adv_x.dim() == 3:
                adv_x.unsqueeze(0)
            # Compute non-fooled target set
            outputs = self.config.model(adv_x)
            if self.config.task == Task.Classification:
                fooled = _compare_misclassification(y, outputs, dim=1)
            elif self.config.task == Task.Segmentation:
                outputs = outputs["out"]
                outputs = outputs.permute(0, 2, 3, 1)
                fooled = _compare_semantic(y, outputs, dim=3)
            else:
                raise NotImplementedError(f"{self.config.task} not available for DAG Attack.")
            not_fooled = torch.nonzero(torch.logical_not(fooled))

            # Early stop if all targets are fooled
            if not_fooled.size(0) == 0:
                # if self.config.verbose:
                #    loop.set_postfix({'early_stopped': True})
                break

            # Indexing for pytorch by column
            idxs = [not_fooled[:, i] for i in range(not_fooled.size(1))]

            # Get the predicted and true labels for not fooled samples
            y_label = torch.argmax(y[*idxs], dim=-1)
            out_label = outputs[*idxs, y_label]
            outputs[*idxs, y_label] = -1  # Avoid picking the same as label (in untargeted)
            y_pred = torch.argmax(outputs[*idxs], dim=-1)
            out_pred = outputs[*idxs, y_pred]

            rs = torch.zeros_like(out_pred)
            # Compute which element should be evaded because the label is negative
            row_mins = torch.amin(y_label, dim=tuple(range(1, y_label.dim())))
            is_targeted = row_mins < 0
            targeted_mask = is_targeted.view(-1, *[1] * (y_label.dim() - 1)).expand_as(y_label)

            # Compute the loss
            rs[targeted_mask] = out_pred[targeted_mask] - out_label[targeted_mask]
            rs[~targeted_mask] = out_label[~targeted_mask] - out_pred[~targeted_mask]
            rs = rs.mean()

            # Optimize the perturbation
            rs += -(1e-1) * torch.nn.CosineSimilarity(dim=0)(perturbation.flatten(), init_perturbation.flatten())
            rs.backward()
            optimizer.step()

            # Project the perturbation on a ball of radius epsilon
            with torch.no_grad():
                self.config.lp_norm.normalize_(perturbation.data)

            # Update description
            # if self.config.verbose:
            #    loop.set_postfix({'not_fooled': not_fooled.size(0), 'rs': rs.item()})

        return perturbation.detach()

    def step(
            self, i: int, x: torch.Tensor, y: Optional[torch.Tensor] = None, ext_results: Optional[dict] = None,
            **kwargs
    ) -> tuple[torch.Tensor, bool]:
        r"""Fast UAP attack step.

        Generate a universal adversarial attack as described in Algorithm 1 of doi.org/10.48550/arXiv.1610.08401.
        integrating observations from "Fast-UAP" doi.org/10.48550/arXiv.1911.01172 of taking into account alignment of new perturbation
        with respect to preceding candidate universal perturbation.
        This specific version use dag attack from `dag.py` underneath, modifying its loss adding a term using the CosineSimilarity.

        :param x: a tensor of shape (B, C, W, H) with B = Number of batches, C = number of channels for each image,
        W = width of the image and H = height of the image.
        :param y: a tensor of shape (B, CLS) with B = Number of batches and CLS being the number of classes for the
        classifier. If targeted, the label to predict.
        :param ext_results: a dictionary storing: * 'iters': integer = number of iterations required before the
        empirical error converges.
        """
        if not hasattr(self, "x_adv"):
            self.x_adv = x.clone()
        if not hasattr(self, "perturbation"):
            self.perturbation = torch.zeros_like(x[0])
        if not hasattr(self, "fooling_ratio"):
            self.fooling_ratio = 0.0

        # Break if the empirical error is high
        if self.fooling_ratio >= 1 - self.config.delta:
            if self.config.verbose:
                print(f"Found valid UAP in {i} iterations.")
            return self.x_adv, True
        # compute the indexes i for which we need to compute delta v_i
        output = self.config.model(self.x_adv)
        not_fooled = torch.nonzero(torch.logical_not(_compare_misclassification(y, output, dim=1))).flatten()
        for j in not_fooled:
            # generate the optimal direction of the minimal perturbation
            new_perturbation = self._generate_aligned_perturbation(
                x=x[j].unsqueeze(0), y=y[j].unsqueeze(0), init_perturbation=self.perturbation
            )
            # Compute the projection on the Ball of radius xi in ell^p norm
            self.perturbation.data = self.config.lp_norm.normalize(self.perturbation + new_perturbation).data

        self.x_adv = x + self.perturbation
        output = self.config.model(self.x_adv)
        self.fooling_ratio = torch.mean(_compare_misclassification(y, output, dim=1) * 1.0)

        if self.config.verbose:
            print(f"Iter {i}: Fooling ratio: {self.fooling_ratio}")
        return self.x_adv, False

    def reset(self):
        super().reset()
        for atr in ["perturbation", "fooling_ratio", "x_adv"]:
            if hasattr(self, atr):
                delattr(self, atr)