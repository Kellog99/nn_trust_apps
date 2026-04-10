from collections.abc import Callable

import torch
from pydantic import Field, field_validator

from nn_trust import Task
from nn_trust.loss._loss import Loss, LossConfig
from nn_trust.loss.loss_factory import LossFactory


class NPSLossConfig(LossConfig):
    color_set: torch.Tensor = Field(
        default=...,
        description="A tensor of shape Nx3, where each row is a pixel triplet allowed color.",
        title="Color Set"
    )
    d2p_map: Callable[[torch.Tensor], torch.Tensor] = Field(
        default_factory=None,
        description="A function that map a tensor image into a tensor of the same shape where the digital color is mapped with the actual produced color.",
        title="D2p Map"
    )

    @field_validator('d2p_map', mode="after")
    def valid_map(cls, v) -> Callable[[torch.Tensor], torch.Tensor]:
        return v if v else lambda x: x


@LossFactory.register(
    name="Non-Printability loss",
    description="The non-printability loss aims to generate colors that are in the color set C. In this way, the attack will result more transferable in the real world.",
    task={Task.Classification, Task.Segmentation, Task.Detection}
)
class NPSLoss(Loss):
    r"""
    The non-printability loss aims to generate colors that are in the color set C. In this way, the attack will result
    more transferable in the real world.

    .. math::

        L_{nps}^{min} = \sum_{p_{patch} \in P}{\min_{c_{print} \in C} |m(p_{patch}) - c_{print}|} \\
        L_{nps}^{sum} = \sum_{p_{patch} \in P}{\prod_{c_{print} \in C}{|m(p_{patch}) - c_{print}|}}

    Those two version differs in the reduction, that can be chosen in the parameters.

    'sum' version: S. Thys, et al., "Fooling automated surveillance cameras: adversarial patches to attack person
    detection" in CoRR abs/1904.08653, 2019.
    'prod' version and color mapping: Sharif, M., et al, "Accessorize to a Crime: Real and Stealthy Attacks on
    State-of-the-Art Face Recognition," in Proceedings of the 2016 ACM SIGSAC Conference on Computer and Communications
    Security, 2016, pp. 1528–1540.
    """

    CONFIG_T = NPSLossConfig

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute the non-printability score of the image x.

        :param x: the image.

        :return: the NPS loss.
        """
        assert x.size() == 3, "Expected 3D tensor (C x H x W)"

        # p=1 specifies the L1 norm (sum of absolute differences)
        distances = torch.cdist(self.config.d2p_map(x), self.config.color_set, p=1)

        if self._reduction not in ["nps_min", "nps_prod"]:
            raise ValueError("The type of reduction is not available. Choose a reduction in ['nps_min', 'nps_prod']")

        return self.reduce(distances)
