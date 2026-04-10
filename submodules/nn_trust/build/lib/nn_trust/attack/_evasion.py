import abc
from contextlib import suppress
from typing import Optional, Type, cast

import torch
import tqdm
from pydantic import BaseModel, Field

from nn_trust.attack.utils._utils import to_device
from nn_trust.attack.utils.logger import Logger
from nn_trust.core import ModelAdapter, Task


class EvasionAttackConfig(BaseModel):
    """
    Default attack config. Make sure that the configs are registered
    """

    model: ModelAdapter = Field(
        default=...,
        description="The model on which to generate the attack."
    )
    task: Task = Field(
        default=...,
        description="The task of the model to attack."
    )
    max_iters: int = Field(
        default=3,
        description="Maximum number of optimization step.",
        ge=1
    )
    p: float = Field(
        default=2.0,
        description="Norm to use for normalising the power of the perturbation.",
        ge=1.0,
    )
    epsilon: float = Field(
        default=1e2,
        description="Force of the attack.",
        gt=0,
    )
    toll: float = Field(
        default=1e-4,
        description="It represents the minimum value that the norm associated with the update of the adversarial perturbation during iterations can take.",
        gt=0.0,
    )

    ####################################### Optional #######################################
    targeted: bool = Field(
        default=False,
        description="It tells whether the attack is targeted or not."
    )
    verbose: bool = Field(
        default=False,
        description="True to generate more debug print."
    )
    device: torch.device = Field(
        default=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        description="The device used both for storing the eventual model and generating the attack.",
    )

    ########################################################################################
    class Config:
        arbitrary_types_allowed = True


class EvasionAttack(abc.ABC):
    """
    Attack abstract class. Make sure that any other attack will be registered


    >>> atk = EvasionAttack(config=config)
    >>> atk.add_variable_to_track("adv_stocas", metadata="image")
    >>> atk.generate(x, y, logger=Logger())
    """

    CONFIG_T: Type[EvasionAttackConfig] = EvasionAttackConfig

    def __init__(
            self,
            config: EvasionAttackConfig,
    ) -> None:
        """Initialize the attack.

        :param config: The attack configuration.
        """
        self._config = cast(self.CONFIG_T, config)
        self._variables_to_track = {}

    @property
    def config(self) -> CONFIG_T:
        return self._config

    def add_variable_to_track(self, name: str, metadata: Optional[str] = None):
        self._variables_to_track[name] = metadata

    def track_variables(self):
        """A function to determine which variables to track that is called a priori before the optimization procedure."""
        self.add_variable_to_track("res", "images")

    def fit(self, proxy_data: torch.utils.data.DataLoader, **kwargs) -> None:
        r"""Implements a fitting procedure for the internal state variables
        of the :class:`EvasionAttack`.

        Consider an algorithm that requires the fitting over a proxy dataset, `proxy_data`, to initialize the internal
        states. This internal state could lead to shortcuts or other improvements in the optimization
        procedure of the :meth:`generate` method.

        :param proxy_data: A :class:`torch.utils.data.DataLoader` containing the data to fit the attack.
        :returns: None
        """
        raise NotImplementedError("`fit` method is not implemented by EvasionAttack.")

    def reset_fit(self):
        r"""Reset potential internal states that were fitted via the :meth:`fit` method."""
        raise NotImplementedError("`reset_fit` method is not implemented by EvasionAttack.")

    def step(
            self,
            i: int,
            x: torch.Tensor,
            y: Optional[torch.Tensor] = None,
            ext_results: Optional[dict] = None,
            **kwargs
    ) -> tuple[torch.Tensor, bool]:
        r"""Implements a step of optimization of for the :meth:`generate` method.

        :param i: iteration step where the generate method is
        :param x: A tensor representing the sample to attack
        :param y: `y` represents the target of an attack. In case of ``Task.Classification`` is a one-hot encoded tensor.
        :param ext_results: A dictionary where further results information are saved.

        It returns the given adversarial :class:`torch.Tensor`.
        """
        raise NotImplementedError("Implement Step method if you want to call it.")

    def reset(self):
        r"""A callback which is called at the end of the optimization procedure defined in :meth:`generate` to remove
        potential internal states of the :meth:`step`."""
        if hasattr(self, "res"):
            delattr(self, "res")

    def generate(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            patch_mask: Optional[torch.Tensor] = None,
            ext_results: Optional[dict] = None,
            logger: Optional[Logger] = None,
            **kwargs,
    ) -> torch.Tensor:
        """Generate an adversarial sample x* starting from a sample x and a target or the original label.

        :param x: A tensor representing the sample to attack
        :param y: `y` represents the target of an attack. In case of ``Task.Classification`` is a one-hot encoded tensor.
        :param patch_mask:
        :param logger:
        :param ext_results: A dictionary where further results information are saved.
        :param ext_results: A dictionary where further results information are saved.
        :return: The resulting tensor of the perturbation p that x* = x + p
        """
        # Creates the logger
        self.logger = logger if logger else Logger()
        loop = tqdm.tqdm(range(1, self._config.max_iters + 1), disable=not self._config.verbose)
        kwargs.update({"loop": loop})
        # automatically tracks the variables
        with suppress(NotImplementedError):
            self.track_variables()

        # Map to the correct device
        to_device(self._config, self._config.device)
        x = x.to(self._config.device)
        if patch_mask is not None:
            patch_mask = patch_mask.to(self._config.device)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        self.logger.log(
            tag="original_images",
            data=x.detach().cpu(),
            state="generate",
            metadata="images"
        )
        y = y.to(self._config.device)
        ###################################################################
        for _i in loop:
            self.res, success = self.step(_i, x, y, patch_mask=patch_mask, ext_result=ext_results, **kwargs)
            self.res = self.res.clone().detach()
            self.res.clamp_(-1.0, 1.0)

            # Automatic logging
            for var_to_track in self._variables_to_track:
                value_to_track = getattr(self, var_to_track)
                self.logger.log(
                    tag=var_to_track,
                    data=value_to_track,
                    metadata=self._variables_to_track[var_to_track],
                    state="generate",
                )
            self.logger.step()
            # NOTE: Early stop by default
            if success:
                break
        results = self.res.clone().detach()
        self.reset()
        return results
