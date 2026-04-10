from typing import Optional

import torch
from pydantic import Field, model_validator

from nn_trust.attack import EvasionAttack, EvasionAttackConfig
from nn_trust.attack import EvasionAttackFactory
from nn_trust.attack.normalization import LpNormalization
from nn_trust.attack.utils._utils import _compare_misclassification
from nn_trust.core import AttackType, Knowledge, Task


class _UAPAttackConfig(EvasionAttackConfig):
    delta: float = Field(
        default=0.1,
        description="Target error rate in classification.",
        ge=0.0,
        le=1.0
    )
    lp_norm: LpNormalization = Field(
        default_factory=lambda: LpNormalization(p=2.0, radius=48.0),
        description="Lp normalization for the perturbation.",
    )
    attack: EvasionAttack | None = Field(
        default=None,
        description="Evasion attack algorithm to reach classification boundary."
    )

    @model_validator(mode="after")
    def validate_sub_attack(self):
        if self.attack is None:
            self.attack = EvasionAttackFactory.create(
                class_id="deepfool",
                model=self.model,
                task=self.task,
                device=self.device
            )
        return self


@EvasionAttackFactory.register(
    name="Universal Adversarial Perturbation",
    description="First example of Universal adversarial perturbation.",
    task={Task.Classification},
    type=AttackType.Digital,
    knowledge=Knowledge.White
)
class _UAPAttack(EvasionAttack):
    """
    Universal Adversarial Perturbation (UAP)
    """

    CONFIG_T = _UAPAttackConfig

    def track_variables(self):
        super().track_variables()
        self.add_variable_to_track("perturbation", "image")

    @torch.no_grad()
    def step(
            self,
            i: int,
            x: torch.Tensor,
            y: Optional[torch.Tensor] = None,
            ext_results: Optional[dict] = None,
            **kwargs
    ) -> tuple[torch.Tensor, bool]:
        r"""
        Generate a universal adversarial attack as described in Algorithm 1 of doi.org/10.48550/arXiv.1610.08401.
        :param i: the step where the iteration is.
        :param x: a tensor of shape (B, C, W, H) with B = Number of batches, C = number of channels for each image,
        W = width of the image and H = height of the image.
        :param y: a tensor of shape (B, CLS) with B = Number of batches and CLS being the number of classes for the
        classifier. If targeted, the label to predict.
        :param ext_results: a dictionary storing: * 'iters': integer = number of iterations required before the
        empirical error converges.
        """
        if not hasattr(self, "x_adv"):
            setattr(self, "x_adv", x.clone())
        if not hasattr(self, "perturbation"):
            setattr(self, "perturbation", torch.zeros_like(x[0]))

        # compute the indexes i for which we need to compute delta v_i
        output = self._config.model(self.x_adv)
        not_fooled = torch.nonzero(torch.logical_not(_compare_misclassification(y, output, dim=1))).flatten()
        for j in not_fooled:
            # generate the optimal direction of the minimal perturbation
            x_adv = self._config.attack.generate(x=x[j].unsqueeze(0) + self.perturbation, y=y[j].unsqueeze(0))
            new_perturbation = x_adv - x[j]
            # Compute the projection on the Ball of radius xi in ell^p norm
            self.perturbation.data = self._config.lp_norm.normalize(self.perturbation + new_perturbation).data

        self.x_adv = x + self.perturbation.detach().clone()
        output = self._config.model(self.x_adv)

        # Break if the empirical error is high
        if _compare_misclassification(y, output, dim=1).float().mean() >= 1 - self._config.delta:
            if self._config.verbose:
                print(f"Found valid UAP in {i} iterations.")
            return self.x_adv, True

        if self._config.verbose:
            print(f"Iter {i}: Fooling ratio: {self.fooling_ratio}")
        return self.x_adv, False

    def reset(self):
        super().reset()
        for atr in ["perturbation", "x_adv"]:
            if hasattr(self, atr):
                delattr(self, atr)
