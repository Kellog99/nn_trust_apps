from typing import Any, Optional, Type

import torch
import torch.optim as optim
from pydantic import Field, field_validator

from nn_trust.attack import EvasionAttackFactory
from nn_trust.attack._evasion import EvasionAttack, EvasionAttackConfig
from nn_trust.attack.normalization import LpNormalization
from nn_trust.attack.utils._utils import _compare_misclassification, _compare_semantic
from nn_trust.core import AttackType, Knowledge, Task


class DAGAttackConfig(EvasionAttackConfig):
    optimizer_type: Type[optim.Optimizer] = Field(
        default=optim.Adam,
        description="The optimizer class to use during the attack."
    )
    optimizer_args: dict[str, Any] = Field(
        default_factory=lambda: dict(lr=0.01, weight_decay=1e-3),
        description="The parameters to use in the optimizer."
    )
    lp_norm: LpNormalization = Field(
        default_factory=lambda: LpNormalization(p=2.0, radius=40.0),
        description="The norm used to normalize with respect to a given ball in the appropriate Lp space with a fixed radius.",
    )

    @field_validator("lp_norm", mode="after")
    def valid_lp_norm(cls, v):
        if not 1.0 <= v.p <= float("inf"):
            raise ValueError("lp_norm must have a norm between 1 and infinity.")
        return v


@EvasionAttackFactory.register(
    name="Dense Adversarial Generation (DAG)",
    task={Task.Classification, Task.Segmentation},
    description="A white-box attack that generates adversarial examples by minimizing the difference between the target prediction and the adversarial output",
    type=AttackType.Digital,
    knowledge=Knowledge.White
)
class DAGAttack(EvasionAttack):
    """
    The original paper is [1]_. Contrarily, the implementation differs from the one described in [1]_.
    At the line 3 on the pseudocode, the paper requires to compute two gradients for each label. Since the label is
    a linear operator, we simplified that parameter computing the gradient only once, after the sum. We also use
    the mean to not generate huge loss values.
    The numerical projection of the loss at line 4 of the pseudocode is managed by projecting the perturbation on
    a ball of radius the norm of the perturbation.

    .. [1] C. Xie, J. Wang, Z. Zhang, Y. Zhou, L. Xie and A. Yuille, "Adversarial Examples for Semantic Segmentation
    and Object Detection," 2017 IEEE International Conference on Computer Vision (ICCV), Venice, Italy, 2017,
    pp. 1378-1387, doi: https://doi.org/10.1109/ICCV.2017.153
    """

    CONFIG_T = DAGAttackConfig

    def track_variables(self):
        super().track_variables()
        self.add_variable_to_track("perturbation", "images")

    def step(
            self,
            i: int,
            x: torch.Tensor,
            y: Optional[torch.Tensor] = None,
            ext_results: Optional[dict] = None,
            **kwargs
    ) -> tuple[torch.Tensor, bool]:

        loop = kwargs.get("loop")
        if not hasattr(self, "perturbation"):
            self.perturbation = torch.zeros_like(x, device=self._config.device, requires_grad=True)

        if not hasattr(self, "optimizer"):
            self.optimizer = self._config.optimizer_type([self.perturbation], **self._config.optimizer_args)

        self.optimizer.zero_grad()
        self.adv_x = x + self.perturbation
        if self.adv_x.dim() == 3:
            self.adv_x.unsqueeze(0)
        # Compute non-fooled target set
        outputs = self._config.model(self.adv_x)
        if self._config.task == Task.Classification:
            self.fooled = _compare_misclassification(y, outputs, dim=1)
        elif self._config.task == Task.Segmentation:
            outputs = outputs["out"]
            outputs = outputs.permute(0, 2, 3, 1)
            self.fooled = _compare_semantic(y, outputs, dim=3)
        else:
            raise NotImplementedError(f"{self._config.task} not available for DAG Attack.")
        not_fooled = torch.nonzero(torch.logical_not(self.fooled))

        # Early stop if all targets are fooled
        if not_fooled.size(0) == 0:
            return self.adv_x, True

        # Indexing for pytorch by column
        idxs = [not_fooled[:, i] for i in range(not_fooled.size(1))]

        # Get the predicted and true labels for not fooled samples
        y_label = torch.argmax(y.abs()[*idxs], dim=-1)
        out_label = outputs[*idxs, y_label]
        outputs[*idxs, y_label] = -1  # Avoid picking the same as label (in untargeted)
        y_pred = torch.argmax(outputs[*idxs], dim=-1)
        out_pred = outputs[*idxs, y_pred]

        # Initialize in case it was not initialized before
        if not hasattr(self, "rs"):
            self.rs = torch.zeros_like(out_pred)
        # Turn to 0 the loss value
        with torch.no_grad():
            self.rs.zero_()

        # Compute which element should be evaded because the label is negative
        row_mins = torch.amin(y_label, dim=tuple(range(1, y_label.dim())))
        is_targeted = row_mins < 0
        targeted_mask = is_targeted.view(-1, *[1] * (y_label.dim() - 1)).expand_as(y_label)

        # Compute the loss
        self.rs[targeted_mask] = out_pred[targeted_mask] - out_label[targeted_mask]
        self.rs[~targeted_mask] = out_label[~targeted_mask] - out_pred[~targeted_mask]
        self.rs = self.rs.mean()

        # Optimize the perturbation
        self.rs.backward()
        self.optimizer.step()

        # Project the perturbation on a ball of radius epsilon
        with torch.no_grad():
            self._config.lp_norm.normalize_(self.perturbation.data)

        # Update description
        if self._config.verbose and loop is not None and hasattr(loop, "set_postfix"):
            loop.set_postfix({"not_fooled": not_fooled.size(0), "rs": self.rs.item()})

        return self.adv_x, False

    def reset(self):
        super().reset()
        for atr in ["perturbation", "adv_x", "rs", "optimizer"]:
            if hasattr(self, atr):
                delattr(self, atr)
