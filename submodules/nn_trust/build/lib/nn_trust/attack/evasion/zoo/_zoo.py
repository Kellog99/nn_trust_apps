from dataclasses import field
from typing import Any, Optional, cast, Literal

import torch
import torch.nn.functional as F
from pydantic import Field
from torch import optim

from nn_trust import AttackType, Knowledge, Task
from nn_trust.attack import EvasionAttack, EvasionAttackConfig, EvasionAttackFactory
from nn_trust.attack.gradient_approximations import (
    GradientEstimator,
    StochasticCoordinateDescentEstimator,
)
from nn_trust.attack.utils._utils import _compare_misclassification


class ZooAttackConfig(EvasionAttackConfig):
    c: float = Field(
        default=0.01,
        description="The loss adversarial regularization.",
        gt=0.0,
        title="Loss' adversarial regularization"
    )

    h: float = Field(
        default=0.0001,
        description="The small movement value to estimate the gradient.",
        gt=0.0,
        lt=1.0,
        title="Derivative's step size"
    )
    tau: float = Field(
        default=0.0,
        description="A value that maximize the transferability.",
        ge=0.0,
        title="Transferability value"
    )
    use_log: bool = Field(
        default=True,
        description="If not, the logits will be used."
    )
    mapping_function: Literal["autoencoder", "bilinear"] = Field(
        default="bilinear",
        description="The function to use to map low dimensional attacked space to original input size.",
    )
    attack_spatial_size: Optional[tuple[int, int]] = Field(
        default=None,
        description="Low dimensional size where we perform the attack"
    )
    autozoom_decoder: Optional[torch.nn.Module] = Field(
        default=None,
        description="Decoder to be used for low-dimensional attack resizing to attacked model input size."
    )
    optimizer_type: type[optim.Optimizer] = Field(
        default=optim.Adam,
        description="Type of the optimizer to use."
    )
    optimizer_args: dict[str, Any] = Field(
        default_factory=lambda: dict(lr=0.1, betas=(0.9, 0.999)),
        description="Optimizer parameters."
    )

    gradient_estimator: type[GradientEstimator] = field(
        default=StochasticCoordinateDescentEstimator,
        metadata=dict(condition=lambda x: x is not None, desc="Type of gradient estimator to be used."),
    )
    gradient_estimator_args: dict[str, Any] = field(
        default_factory=lambda: dict(n_samples=8), metadata=dict(desc="Parameters for the gradient estimator.")
    )


@EvasionAttackFactory.register(
    id="zoo_method",
    name="Zeroth Order Optimization Method",
    description="Black-box attack leverages efficient gradient estimation through finite differences with additional dimensionality reduction techniques.",
    task={Task.Classification},
    type=AttackType.Digital,
    knowledge=Knowledge.White
)
class _ZooAttack(EvasionAttack):
    r"""
    This class implement Autozoom attack from https://doi.org/10.48550/arXiv.1805.11770
    """

    CONFIG_T = ZooAttackConfig

    def __init__(self, config: EvasionAttackConfig):
        super().__init__(config)
        self._config = cast(ZooAttackConfig, self._config)

    def _upscale_perturbation(self, low_dim_perturbation, input):
        if self._config.mapping_function == "bilinear" and self._config.attack_spatial_size:
            upscaled_perturbation = F.interpolate(low_dim_perturbation, size=input.shape[-2:], mode="bilinear")
        elif (
                self._config.mapping_function == "autoencoder"
                and self._config.autozoom_decoder
                and self._config.attack_spatial_size
        ):
            upscaled_perturbation = self._config.autozoom_decoder(low_dim_perturbation)
        else:
            upscaled_perturbation = low_dim_perturbation
        return upscaled_perturbation

    @torch.no_grad()
    def _adv_loss(self, pred_result: torch.Tensor, true_labels: torch.Tensor):
        """
        Function compute adversarial loss
        """
        t_index = torch.argmax(true_labels.abs(), dim=-1)
        rows = torch.arange(pred_result.size(0))
        pred_result_c = pred_result.clone()
        score_target = pred_result[rows, t_index].clone()
        pred_result_c[rows, t_index] = pred_result.min() - 0.01  # So this can not be the maximum
        score_other = pred_result_c.max(1).values
        if self._config.use_log:
            score_target = torch.log(score_target)
            score_other = torch.log(score_other)

        row_mins = torch.amin(true_labels, dim=tuple(range(1, true_labels.dim())))
        is_targeted = row_mins < 0

        adv_loss = torch.zeros_like(score_target)
        if is_targeted.any():
            adv_loss[is_targeted] = torch.max(
                score_other[is_targeted] - score_target[is_targeted], torch.tensor(-self._config.tau)
            )
        if (~is_targeted).any():
            adv_loss[~is_targeted] = torch.max(
                score_target[~is_targeted] - score_other[~is_targeted], torch.tensor(-self._config.tau)
            )
        # Reduce the loss to a value per batch item
        adv_loss = adv_loss.mean(dim=0)
        return adv_loss

    @torch.no_grad()
    def _zoo_loss(self, perturbation, predicted, y) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        l2_loss = perturbation.view(perturbation.size(0), -1).norm(dim=-1, p=2.0).mean(dim=0)
        adv_loss = self._adv_loss(predicted, y)
        loss = (l2_loss + self._config.c * adv_loss).sum()
        return loss, l2_loss, adv_loss

    @torch.no_grad()
    def _attack_loss(self, x, y, perturbation) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        result = self._config.model(x + perturbation)
        if self._config.use_log:
            result = F.softmax(result, -1)
        loss_vals = self._zoo_loss(perturbation, result, y)
        return loss_vals

    def track_variables(self):
        super().track_variables()
        self.add_variable_to_track("perturbation", "images")
        self.add_variable_to_track("loss", "scalar")
        self.add_variable_to_track("l2_loss", "scalar")
        self.add_variable_to_track("adv_loss", "scalar")

    @torch.no_grad()
    def step(
            self, i: int, x: torch.Tensor, y: Optional[torch.Tensor] = None, ext_results: Optional[dict] = None,
            **kwargs
    ) -> tuple[torch.Tensor, bool]:
        # tqdm and verbose logging
        loop = kwargs.get("loop")
        log_on_tqdm = self._config.verbose and loop is not None and hasattr(loop, "set_postfix")

        n_query = 0
        # The initial adversarial perturbation is 0
        attack_size = list(x.shape)
        if self._config.attack_spatial_size:
            attack_size[-2:] = self._config.attack_spatial_size

        if not hasattr(self, "perturbation"):
            self.perturbation = torch.zeros(size=attack_size).to(self._config.device)
        if not hasattr(self, "x_adv"):
            self.x_adv = x.clone()
        if not hasattr(self, "gradient_estimator"):
            self.gradient_estimator = self._config.gradient_estimator(**self._config.gradient_estimator_args)
        if not hasattr(self, "optimizer"):
            self.optimizer = self._config.optimizer_type([self.perturbation], **self._config.optimizer_args)
        if not hasattr(self, "iter_params"):
            self.iter_params = {
                "n_query": 0,
                "is_success": False,
                "prediction": None,
            }
        # early stop!
        if hasattr(self, "iter_params") and self.iter_params["is_success"]:
            return self.x_adv, True

        def loss_func(perturbation_):
            return self._attack_loss(x=x, y=y, perturbation=self._upscale_perturbation(perturbation_, x))[0]

        def closure():
            perturbation_fullsize = self._upscale_perturbation(self.perturbation, x)
            grad = self.gradient_estimator.gradient(loss=loss_func, params=self.perturbation)
            n_query_grad = self.gradient_estimator.n_query
            self.perturbation.grad = grad
            result = self._config.model(x + perturbation_fullsize)
            if self._config.use_log:
                result = F.softmax(result, -1)
            loss, l2_loss, adv_loss = self._zoo_loss(perturbation_fullsize, result, y)
            self.iter_params["n_query"] += n_query_grad + 1
            self.iter_params["is_success"] = torch.all(_compare_misclassification(y, result, dim=-1))
            self.l2_loss = l2_loss.mean().item()
            self.adv_loss = adv_loss.mean().item()
            self.iter_params["result"] = result
            return loss

        self.loss = self.optimizer.step(closure)
        self.loss = self.loss.item()
        predicted_label = torch.argmax(self.iter_params["result"][0])

        if log_on_tqdm:
            loop.set_postfix(
                {
                    "n_quey": self.iter_params["n_query"],
                    "attack found": self.iter_params["is_success"],
                    "l2-loss": self.l2_loss,
                    "adv-loss": self.adv_loss,
                    "classified": predicted_label.item(),
                    "score": self.iter_params["result"][0, predicted_label].item(),
                }
            )

        self.x_adv = x + self._upscale_perturbation(self.perturbation, x)
        return self.x_adv, False

    def reset(self):
        super().reset()
        for atr in [
            "x_adv",
            "perturbation",
            "gradient_estimator",
            "optimizer",
            "iter_params",
            "loss",
            "l2_loss",
            "adv_loss"

        ]:
            if hasattr(self, atr):
                delattr(self, atr)
