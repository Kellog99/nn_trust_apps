from abc import ABC
from typing import TypeVar, Generic

import torch
from pydantic import BaseModel, Field
from torchmetrics import Metric


class StatisticConfig(BaseModel):
    # I have to put a string otherwise Pydantic cannot validate it.
    device: torch.device = Field(
        default=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        description="The device used both for storing the eventual model and generating the attack.",
        title="Device"
    )
    num_classes: int = Field(
        default=10,
        description="Number of classes for the classification task.",
        title="Number of Classes"
    )
    p: float = Field(
        default=2.0,
        description="Norm to use for normalising the power of the perturbation.",
        ge=1.0,
        title="Order norm"
    )

    class Config:
        arbitrary_types_allowed = True


ConfigT = TypeVar('ConfigT', bound='StatisticConfig')


class Statistic(Metric, ABC, Generic[ConfigT]):
    """
    Base statistic abstract class.
    Make sure that any other attack will be registered
    """
    CONFIG_T = StatisticConfig

    def __init__(
            self,
            config: ConfigT,
    ) -> None:
        """Initialize the attack.

        :param config: The attack configuration.
        """
        super().__init__()
        self._config = config

    @property
    def config(self) -> ConfigT:
        return self._config
