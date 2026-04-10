from typing import Literal, Optional

import torch
from pydantic import Field

from nn_trust.attack._evasion import EvasionAttack, EvasionAttackConfig
from nn_trust.attack.attack_factory import  EvasionAttackFactory
from nn_trust.attack.utils._utils import _compare_misclassification
from nn_trust.core import AttackType, Knowledge, Task


class DUAttackConfig(EvasionAttackConfig):
    epsilon: float = Field(
        default=0.2,
        description="The attack step size (lr like).",
        ge=0.0,
        le=1.0
    )

    delta: float = Field(
        default=300.0,
        description="The maximum l2 norm of the perturbation.",
        gt=0.0,
        lt=float('inf')
    )

    mode: Literal["eye", "rand"] = Field(
        default="rand",
        description="'eye' makes a random permutation of the eye matrix, 'rand' works on random orthogonal columns",
    )


@EvasionAttackFactory.register(
    name="Decision Based Universal Adversarial Attack",
    description="A black-box attack generating an universal adversarial perturbation by step-by-step 'greedy' improvements.",
    task={Task.Classification},
    type=AttackType.Digital,
    knowledge=Knowledge.Black
)
class DUAttack(EvasionAttack):
    CONFIG_T = DUAttackConfig

    def _shift_eye(self, height: int, width: int, mode: Literal["eye", "rand"]) -> torch.Tensor:
        if mode == "eye":
            # Calculate the indices for the shifted and wrapped columns
            k = torch.randint(0, width, (1,))
            indices = (torch.arange(width, device=self._config.device) + k) % width
        elif mode == "rand":
            indices = torch.randperm(width, device=self._config.device)
        # Return the reindex matrix
        return torch.eye(max(width, height), device=self._config.device)[:height, indices]

    def track_variables(self):
        super().track_variables()
        self.add_variable_to_track("perturbation", "image")
        self.add_variable_to_track("momentum", "image")

    @torch.no_grad()
    def step(
            self,
            i: int,
            x: torch.Tensor,
            y: Optional[torch.Tensor] = None,
            ext_results: Optional[dict] = None,
            **kwargs
    ) -> tuple[torch.Tensor, bool]:
        loop = kwargs.get("loop")
        # Start 0 UAP
        if not hasattr(self, "perturbation"):
            self.perturbation = torch.zeros_like(x[0], device=self._config.device)
        if not hasattr(self, "x_adv"):
            self.adv_x = x.clone()
        # Initialize momentum
        if not hasattr(self, "momentum"):
            self.momentum = torch.zeros_like(x[0], device=self._config.device)

        batch_size, n_channels, height, width = x.size()
        # Select the c-th channel and the k-th eye matrix randomly
        c = torch.randint(0, n_channels, (1,))
        q_mask = torch.zeros((n_channels, height, width), device=self._config.device)
        q_mask[c, ...] = self._shift_eye(height, width, self._config.mode)
        perturbation_l = self.perturbation - (self._config.epsilon + 0.9 * self.momentum) * q_mask
        perturbation_l = self._config.delta * perturbation_l / torch.norm(perturbation_l, p="fro")

        output_l = self._config.model((x - perturbation_l).clip(-1, 1))
        re_l = _compare_misclassification(y, output_l, dim=1)
        if torch.all(re_l):
            self.perturbation = perturbation_l
            self.adv_x = x + self.perturbation.clip(-1, 1)
            return self.adv_x, True

        perturbation_r = self.perturbation + (self._config.epsilon + 0.9 * self.momentum) * q_mask
        perturbation_r = self._config.delta * perturbation_r / torch.norm(perturbation_r, p="fro")

        output_r = self._config.model((x - perturbation_r).clip(-1, 1))
        re_r = _compare_misclassification(y, output_r, dim=1)
        if torch.all(re_r):
            self.perturbation = perturbation_r
            self.adv_x = (x + self.perturbation).clip(-1, 1)
            return self.adv_x, True

        if torch.count_nonzero(re_l) > torch.count_nonzero(re_r):
            self.momentum = self.momentum + self._config.epsilon * q_mask
            self.perturbation = perturbation_l
        else:
            self.momentum = self.momentum - self._config.epsilon * q_mask
            self.perturbation = perturbation_r

        if self._config.verbose and loop is not None and hasattr(loop, "set_postfix"):
            loop.set_postfix({"re_l": torch.count_nonzero(re_l), "re_r": torch.count_nonzero(re_r)})

        self.adv_x = (x + self.perturbation).clip(-1, 1)
        return self.adv_x, False

    def reset(self):
        super().reset()
        for atr in [
            "adv_x",
            "perturbation",
            "momentum"
        ]:
            if hasattr(self, atr):
                delattr(self, atr)
