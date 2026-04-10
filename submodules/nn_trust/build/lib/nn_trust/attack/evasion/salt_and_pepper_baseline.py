from contextlib import suppress
from typing import Optional

import torch
from pydantic import Field
from tqdm.auto import tqdm

from nn_trust import AttackType, Knowledge, Task
from nn_trust.attack import EvasionAttack, EvasionAttackConfig, EvasionAttackFactory
from nn_trust.attack.utils._utils import to_device
from nn_trust.attack.utils.logger import Logger


class SaltNPepperNoiseBaselineAttackConfig(EvasionAttackConfig):
    max_perturb: float = Field(
        default=2.0,
        description="Value to set the salt or pepper value",
        ge=0.0,
        title="Max perturbation"
    )


@EvasionAttackFactory.register(
    name="Salt and Pepper Noise",
    description="A black-box attack that adds a salt and pepper noise to the image.",
    task={Task.Classification, Task.Detection, Task.Segmentation},
    type=AttackType.Digital,
    knowledge=Knowledge.Black
)
class SaltNPepperNoiseBaselineAttack(EvasionAttack):
    r"""Generates a `Salt and Pepper type of Noise
    <https://en.wikipedia.org/wiki/Salt-and-pepper_noise>`_ with the `strength`
    being a parameter on the probability that a pixel goes 'dark' on a
    channel.
    """

    CONFIG_T = SaltNPepperNoiseBaselineAttackConfig

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

        pert = torch.rand_like(x)
        pixels_to_modify = pert > self._config.epsilon
        signs = (pert - 0.5).sign()
        self.res = torch.where(pixels_to_modify, x, self._config.max_perturb * signs)
        self.perturbation = torch.where(pixels_to_modify, 0.0, self._config.max_perturb * signs)
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
