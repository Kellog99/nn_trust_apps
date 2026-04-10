from typing import Optional

import torch
from pydantic import Field, field_validator

from nn_trust.attack import EvasionAttack, EvasionAttackConfig, EvasionAttackFactory
from nn_trust.attack.sampling_utils import CartesianSampler, DCTSampler, Sampler
from nn_trust.attack.utils._utils import _compare_misclassification
from nn_trust.core import AttackType, Knowledge, Task


class SimBAAttackConfig(EvasionAttackConfig):
    epsilon: float = Field(
        default=0.33,
        description="Value of the exponential for determining the optimal queries per iteration",
        gt=0.0,
        title="Exponential number for queries generation"
    )

    basis_generator: type[Sampler] = Field(
        default="dct",
        description="Method used to generate the random directions",
        validate_default=True
    )

    @field_validator("basis_generator", mode="before")
    def basis_generator_validator(cls, v):
        if isinstance(v, str):
            if v == "dct":
                v = DCTSampler
            elif v == "cartesian":
                v = CartesianSampler
        return v


@EvasionAttackFactory.register(
    name="Simple Black Box Attack (SimBA)",
    description="A black-box attack that greedily decide a cosine perturbation to add to the image aimed at misclassify.",
    task={Task.Classification},
    type=AttackType.Digital,
    knowledge=Knowledge.Black
)
class SimBAAttack(EvasionAttack):
    r"""Implements a black-box adversarial attack based on a very simple procedure: either you add noise or remove noise.
    The core idea is to find a suitable basis :math:`\mathcal{B}`, select random noise :math:`v \in \mathcal{B}`
    and check the optimal direction between :math:`-\varepsilon v` and :math:`\varepsilon v`. The goal is to minimize
    the logit response of the passed neural network model with respect to the original prediction class.
    Additional details to improve this technique can be found in [1]_.

    :class:`SimBAAttackConfig` provides parameters used for the configuration of the attack that are passed as
    a ``config`` variable in the initialized of :class:`SimBAAttack`.

    :param config: The configuration can be tweaked via changing the following parameters:
        - ``epsilon`` which specifies updating step of the optimization procedure.
        - ``basis_generator`` the type of basis to sample the random directions from. Default is 'dct'.
        - ``max_iters`` the maximum number of iterations of the optimization procedure.

    Example::

    Consider a batch of images :math:`(B, C, H, W)`, which we denote as ``DATA_INPUT``, and a corresponding
    one-hot encoded labels ``TARGET_LABEL`` of shape :math:`(B, N_c)`. Let ``MODEL`` be a :class:`ModelAdapter`.
    Then, we can use the attack as follows

    >>> from nn_trust.attack import EvasionAttackFactory
    >>> cnf = EvasionAttackFactory.get_config('simba',
    >>>         model=MODEL,
    >>>         task=Task.Classification,
    >>>         basis_generator='dct',
    >>>         max_iters=100)
    >>> atk = EvasionAttackFactory.create_attack(config=cnf)
    >>> atk.generate(DATA_INPUT, TARGET_LABEL)

    .. [1] Guo, Chuan, Jacob R. Gardner, Yurong You, Andrew Gordon Wilson and Kilian Q. Weinberger.
        “Simple Black-box Adversarial Attacks.” ArXiv abs/1905.07121 (2019) https://arxiv.org/abs/1905.07121.
    """

    CONFIG_T = SimBAAttackConfig

    def track_variables(self):
        super().track_variables()
        self.add_variable_to_track("perturbation", "images")
        self.add_variable_to_track("remaining_indices", "tensor")

    @torch.no_grad()
    def step(
            self, i: int, x: torch.Tensor, y: Optional[torch.Tensor] = None, ext_results: Optional[dict] = None,
            **kwargs
    ) -> tuple[torch.Tensor, bool]:
        r"""Generates a value ``x_adv`` such that it is misclassified by the configuration model.

        :param x: a tensor of shape :math:`(B, C, H, W)`, respectively the batch size, number of channels,
            width and height of the image.
        :param y: a tensor of shape :math:`(B, N_classes)` with :math:`N_\text{classes}` being the number of
            classes the classifier is able to classify.

        :raise AttackException: if ``y`` is ``None``; either a target or the original label is required.

        :returns: a :class:`torch.Tensor` with same size of ``x``.
        """
        if not hasattr(self, "x_adv"):
            self.x_adv = x.clone()

        if not hasattr(self, "perturbation"):
            self.perturbation = self.x_adv - x

        # Initialize a sampler wrt the configuration choice.
        if not hasattr(self, "sampler"):
            self.sampler = self._config.basis_generator(x.shape[1:], device=self._config.device)
            # Restrict the sampling to low frequency only
            if type(self.sampler) is DCTSampler:
                self.sampler._max_freq = (x.numel() / x.size(0) / x.size(1)) / 2
        # Create masks and indices for reducing the batch size and allow vectorized code.
        if not hasattr(self, "remaining_indices"):
            self.remaining_indices = torch.ones(x.shape[0], dtype=bool, device=self._config.device)
        if not hasattr(self, "batch_indices"):
            self.batch_indices = torch.arange(0, x.shape[0], dtype=torch.long, device=self._config.device)
        # Take the probabilities for the given classes.
        if not hasattr(self, "est_probs"):
            self.est_probs = self._config.model(x)[self.batch_indices, y.argmax(dim=-1)]

        # Early stopping
        if hasattr(self, "remaining_indices") and torch.all(torch.logical_not(self.remaining_indices)):
            if self._config.verbose:
                print(f"### EARLY STOPPING AT ITERATION={i}.")
            return self.x_adv, True

        # Mask the batch to the elements left.
        # TODO: implement it a bit better to account for targeted/ untargeted
        tmp_x_adv = self.x_adv[self.remaining_indices]
        tmp_probs = self.est_probs[self.remaining_indices]
        labels_left_onehot = y[self.remaining_indices, :]
        labels_left = labels_left_onehot.argmax(dim=-1)
        batch_indices_left = torch.arange(0, tmp_x_adv.shape[0], dtype=torch.long, device=self._config.device)
        random_sample = self.sampler.sample(n=int(self.remaining_indices.sum()))
        # Computes the model evaluation wrt  + eps * adversarial sample
        left_vecs = tmp_x_adv + random_sample * self._config.epsilon
        left_model_evals = self._config.model(left_vecs)[batch_indices_left, labels_left]
        left_vecs_are_better = left_model_evals.lt(tmp_probs)
        if torch.any(left_vecs_are_better):
            tmp_x_adv[left_vecs_are_better] = left_vecs[left_vecs_are_better]
            tmp_probs[left_vecs_are_better] = left_model_evals[left_vecs_are_better]

        # Computes the model evaluation wrt  - eps * adversarial sample
        right_vecs = tmp_x_adv - random_sample * self._config.epsilon
        right_model_evals = self._config.model(right_vecs)[batch_indices_left, labels_left]
        right_vecs_are_better = right_model_evals.lt(tmp_probs)
        if torch.any(right_vecs_are_better):
            tmp_x_adv[right_vecs_are_better] = right_vecs[right_vecs_are_better]
            tmp_probs[right_vecs_are_better] = right_model_evals[right_vecs_are_better]

        # Update the probs tensor and the adversarial output
        self.est_probs[self.remaining_indices] = tmp_probs
        # Clamp the image back to the correct scale
        self.x_adv[self.remaining_indices] = tmp_x_adv.clamp(min=-1.0, max=1.0)

        # Check if we have improvements regarding mis-classification
        correctly_classified = _compare_misclassification(
            labels_left_onehot, self._config.model(tmp_x_adv)
        ).logical_not()
        self.remaining_indices[self.remaining_indices.clone()] = correctly_classified
        if self._config.verbose:
            print("Number of correctly classified elements: ", self.remaining_indices.sum())

        self.perturbation = self.x_adv - x

        return self.x_adv, False

    def reset(self):
        super().reset()
        for atr in [
            "est_probs",
            "remaining_indices",
            "batch_indices",
            "sampler",
            "x_adv",
            "perturbation"
        ]:
            if hasattr(self, atr):
                delattr(self, atr)
