from contextlib import suppress
from typing import Optional

import torch
from pydantic import Field
from tqdm.auto import tqdm

from nn_trust import AttackType, Knowledge, Task
from nn_trust.attack import EvasionAttack, EvasionAttackConfig, EvasionAttackFactory
from nn_trust.attack.normalization import LpNormalization
from nn_trust.attack.utils._utils import to_device
from nn_trust.attack.utils.logger import Logger


class GaussianBaselineAttackConfig(EvasionAttackConfig):
    pass


@EvasionAttackFactory.register(
    name="Gaussian Baseline",
    description="A black-box attack adding a random gaussian value to each pixel.",
    task={Task.Classification},
    type=AttackType.Digital,
    knowledge=Knowledge.Black
)
class GaussianBaselineAttack(EvasionAttack):
    r"""Generates a simple attack by adding a randomly sampled normal value to
    each pixel.
    """

    CONFIG_T = GaussianBaselineAttackConfig

    def track_variables(self):
        """A function to determine which variables to track that is called a priori before the optimization procedure."""
        super().track_variables()
        self.add_variable_to_track("perturbation", "images")

    def generate(
            self,
            x: torch.Tensor,
            y: torch.Tensor = None,
            ext_results: Optional[dict] = None,
            **kwargs,
    ) -> torch.Tensor:
        # Creates the logger
        self.logger = kwargs.get("logger", Logger())
        loop = tqdm(range(1, self._config.max_iters + 1), disable=not self._config.verbose)
        kwargs.update({"loop": loop})
        # automatically tracks the variables
        with suppress(NotImplementedError):
            self.track_variables()

        # Map to the correct device
        to_device(self._config, self._config.device)
        x = x.to(self._config.device)
        if x.dim() == 3:
            x = x.unsqueeze(0)

        self.logger.log(tag="original_images", data=x.detach().cpu(), state="generate", metadata="images")
        self.logger.log(
            tag="original_classification",
            data=y.abs().argmax(dim=-1).detach().cpu(),
            state="generate",
            metadata="tensor",
        )
        y = y.to(self._config.device)

        normalizer = LpNormalization(p=self._config.p, radius=self._config.epsilon, center=x)
        self.perturbation = normalizer(x + torch.randn_like(x) * self._config.epsilon)
        self.res = x + self.perturbation
        self.logits = self._config.model(self.res).detach().cpu()
        self.probs = self.logits.softmax(dim=-1).detach().cpu()
        self.model_adv_classification = self.probs.argmax(dim=-1)
        # Automatic logging
        for var_to_track in self._variables_to_track:
            value_to_track = getattr(self, var_to_track)
            self.logger.log(
                tag=var_to_track,
                data=value_to_track,
                metadata=self._variables_to_track[var_to_track],
                state="generate",
            )

        return self.res

    def reset(self):
        super().reset()
        del self.perturbation
