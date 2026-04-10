import abc
from typing import Optional

import torch
from pydantic import Field

from nn_trust.attack._evasion import EvasionAttack, EvasionAttackConfig
from nn_trust.attack.utils._utils import _compare_misclassification
from nn_trust.core import AttackType, Knowledge, Task


class _TensorBuffer:
    def __init__(
            self,
            max_size: int,
            device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            dtype=torch.float32,
    ):
        """
        Args:
            max_size (int): Maximum number of elements to store.
            device (str): Device to store the tensor.
            dtype (torch.dtype): Data type of the tensor.
        """
        self.max_size = max(max_size, 1)
        self.device = device
        self.dtype = dtype
        self.buffer = torch.empty((0,), device=device, dtype=dtype)  # Initialize empty buffer

    def append(self, value):
        """Add a new value to the buffer."""
        value = value.to(self.device).unsqueeze(0)  # Make it (1, value_shape)
        self.buffer = torch.cat([self.buffer, value])  # Add to the end
        if len(self.buffer) > self.max_size:
            self.buffer = self.buffer[1:]  # Remove the first element to maintain size

    def get(self):
        """Return the current buffer."""
        return self.buffer

    def get_differences(self):
        if len(self.buffer) < self.max_size:
            return torch.tensor(float("inf"))
        else:
            return self.buffer.diff().abs().max()


class _BoundaryAttackConfig(EvasionAttackConfig):
    initial_point: torch.Tensor = Field(
        default=None,
        description="The initial adversarial point to start the procedure."
    )
    boundary: tuple[float, float] = Field(
        default=(-1.0, 1.0),
        description="Boundary of the model's domain"
    )
    history: int = Field(
        default=10,
        description="Number of steps to look in the past for stating if the loss is changing or not.",
        gt=0,
        title="History"
    )


class _BoundaryAttack(EvasionAttack):
    CONFIG_T = _BoundaryAttackConfig
    TASKS = {Task.Classification}
    ATTACK_TYPE = AttackType.Digital
    ATTACK_KNOWLEDGE = Knowledge.Black

    def track_variables(self):
        super().track_variables()
        self.add_variable_to_track("perturbation", "images")
        self.add_variable_to_track("is_adversarial", "tensor")

    @abc.abstractmethod
    def get_movement(self, x: torch.Tensor, x_adv: torch.Tensor) -> torch.Tensor:
        """
        This method has to generate the new direction
        :param x: the original image.
        :param x_adv: the adversarial input at the previous step.
        """

    @abc.abstractmethod
    def update_parameters(self, buffer_parameters: _TensorBuffer, is_adversarial: torch.Tensor, **kwargs) -> None:
        """
        This method has to update the step parameters in case certain conditions occur
        """

    def step(
            self, i: int, x: torch.Tensor, y: Optional[torch.Tensor] = None, ext_results: Optional[dict] = None,
            **kwargs
    ) -> tuple[torch.Tensor, bool]:
        ############### STARTING POINT ###############
        # The starting point is essential
        # by construction it must be already an adversarial input
        # N(x_adv) != N(x)
        if not hasattr(self, "x_adv"):
            # Generating a noisy data that belongs in the domain
            self.x_adv = self._config.initial_point if self._config.initial_point else x + torch.randn_like(x) * 0.1
            self.x_adv = torch.clamp(self.x_adv, min=self._config.boundary[0], max=self._config.boundary[1])
        if not hasattr(self, "perturbation"):
            self.perturbation = torch.zeros_like(x)
        if not hasattr(self, "buffer_stop"):
            self.buffer_stop = _TensorBuffer(max_size=self._config.history)
        if not hasattr(self, "buffer_parameters"):
            self.buffer_parameters = _TensorBuffer(max_size=self._config.history // 2)
        if not hasattr(self, "dist"):
            self.dist = []

        y = y if y is not None else self._config.model(x)



        if i <= 1:
            # Tells which input satisfy the adversarial conditions.
            # if the attack is targeted then it is needed to change the element that are not the same as y
            # if the attack is untargeted then it is needed to change the element that are the same as the prediction
            self.out = self._config.model(self.x_adv).argmax(-1)
            self.is_adversarial = _compare_misclassification(
                y_label=y,
                y_pred=self.out,
                reduction="none"
            )
            not_adversarial = torch.logical_not(self.is_adversarial)
            while torch.any(not_adversarial):
                # Now I want to change all the inputs that are not adversarial
                tmp = x[not_adversarial] + torch.randn_like(x[not_adversarial])
                tmp = torch.clamp(tmp, min=self._config.boundary[0], max=self._config.boundary[1]).to(
                    self._config.device
                )

                self.x_adv[not_adversarial] = tmp
                self.out[not_adversarial] = self._config.model(tmp).argmax(-1)

                self.is_adversarial = _compare_misclassification(
                    y_label=y,
                    y_pred=self.out,
                    reduction="none"
                )
                not_adversarial = torch.logical_not(self.is_adversarial)

            # At this time `x_adv` is a batch of adversarial perturbation
            self.perturbation = self.x_adv - x
            distance = torch.norm(self.perturbation.flatten(1), dim=-1, p=self._config.p)
            self.proximity = distance < self._config.toll
            if torch.all(self.proximity):
                # I can already interrupt the procedure
                return self.x_adv, True

            self.buffer_stop.append(distance.max())
            self.buffer_parameters.append(distance.max())
            self.dist.append(distance.max().item())
        else:
            not_proximal = torch.logical_not(self.proximity)
            self.candidate = self.get_movement(x=x[not_proximal], x_adv=self.x_adv[not_proximal])

            # Clip to valid pixel range
            self.candidate = torch.clamp(self.candidate, min=self._config.boundary[0], max=self._config.boundary[1])

            self.is_adversarial = _compare_misclassification(
                y_label=y[not_proximal],
                y_pred=self._config.model(self.candidate).argmax(-1),
                reduction="none",
            )
            # apply the Accept-rejection statement
            self.x_adv_tmp = self.x_adv[not_proximal]
            self.x_adv_tmp[self.is_adversarial] = self.candidate[self.is_adversarial]
            self.x_adv[not_proximal] = self.x_adv_tmp

            # parameters update
            self.update_parameters(self.buffer_parameters, self.is_adversarial)

            # update the stop condition
            distance = torch.norm((self.x_adv - x).flatten(1), dim=1, p=self._config.p)
            tmp_proximity = distance < self._config.toll
            self.proximity = torch.logical_or(tmp_proximity, self.proximity)

            self.buffer_stop.append(distance.max())
            self.buffer_parameters.append(distance.max())
            self.dist.append(distance.max().item())
            if torch.logical_or(torch.all(self.proximity), self.buffer_stop.get_differences() < self._config.toll):
                # The adversarial perturbation is pretty close to the original image
                # Hence, it is possible to stop the iterations
                self.perturbation = self.x_adv - x
                return self.x_adv, True

        self.perturbation = self.x_adv - x
        return self.x_adv, False

    def reset(self):
        super().reset()
        for atr in ["perturbation", "buffer_stop", "x_adv", "buffer_parameters", "x_adv_tmp", "candidate", "proximity",
                    "out"]:
            if hasattr(self, atr):
                delattr(self, atr)