from typing import Literal, TypeVar, Generic

import torch
from pydantic import BaseModel, Field


class LossConfig(BaseModel):
    # I have to put a string otherwise Pydantic cannot validate it.
    device: torch.device = Field(
        default=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        description="The device used both for storing the eventual model and generating the attack.",
        title="Device"
    )
    p: float = Field(
        default=2.0,
        description="Norm to use for normalising the power of the perturbation.",
        ge=1.0,
        title="Order norm"
    )
    reduction: Literal["none", "sum", "mean", "max", "batch_mean", "min", "nps_min", "nps_prod"] = Field(
        default="mean",
        description="Type of reduction to do on the loss.",
        title="Reduction"
    )

    class Config:
        arbitrary_types_allowed = True


ConfigT = TypeVar('ConfigT', bound='LossConfig')


class Loss(torch.nn.Module, Generic[ConfigT]):
    """
    Loss abstract class.
    Make sure that any other attack will be registered
    """
    CONFIG_T: type[LossConfig] = LossConfig

    def __init__(self, config: LossConfig) -> None:
        super(Loss, self).__init__()
        self._config = config

    @property
    def config(self) -> ConfigT:
        return self._config

    def reduce(self, x: torch.Tensor) -> torch.Tensor:
        """
        It contains all the possible reduction of the losses that a loss can use.

        :param x: is at least a 2D tensor
        """

        if self.config.reduction == "none":
            return x

        elif self.config.reduction == "sum":
            return x.sum()

        elif self.config.reduction == "mean":
            return x.mean()

        elif self.config.reduction == "batch_mean":
            return x.sum() / x.shape[0]

        elif self.config.reduction == "max":
            return x.max()

        elif self.config.reduction == "min":
            return x.min()

        elif self.config.reduction == "nps_prod":
            return x.prod(dim=1).sum()

        elif self.config.reduction == "nps_min":
            return x.min(dim=1).sum()

        else:
            raise ValueError(f"The reduction {self.config.reduction} is not available.")
