from typing import Optional, cast

import torch
from pydantic import Field
from torch import optim

from nn_trust import AttackType, Knowledge, Task
from nn_trust.attack._evasion import EvasionAttack, EvasionAttackConfig
from nn_trust.attack.attack_factory import EvasionAttackFactory
from nn_trust.attack.normalization import LpNormalization
from nn_trust.attack.utils._utils import _compare_misclassification
from nn_trust.loss.loss_composer import LossComposer


class BanditPriorAttackConfig(EvasionAttackConfig):
    delta: float = Field(
        default=0.1,
        description="Bandit exploration step size.",
        ge=0.0,
        lt=float('inf')
    )

    eps: float = Field(
        default=0.1,
        description="Finite difference step size.",
        ge=0.0,
        lt=float('inf')
    )

    loss: torch.nn.Module = Field(
        default_factory=lambda: LossComposer(losses={"misclassification": {}}),
        description="The loss function to use for the attack.",
    )

    optimizer: type[optim.Optimizer] = Field(default=optim.SGD, description="Optimizer to use.")

    perturbation_optimizer_params: dict[str, float | int] = Field(
        default_factory=lambda: dict(lr=0.1),
        description="Parameters of the optimizer."
    )
    state_optimizer_params: dict[str, float | int] = Field(
        default_factory=lambda: dict(lr=0.1),
        description="Parameters of the optimizer."
    )
    epsilon: float = Field(
        default=80.0,
        description="Perturbation strength expressed as perturbation norm, using the given Lp norm.",
        gt=0.0,
        lt=float('inf')
    )


@EvasionAttackFactory.register(
    name="Bandit and Prior.",
    description="A black-box adversarial attack deemed to estimate the loss gradient via Bandit optimization to find an optimal perturbation.",
    task={Task.Classification},
    type=AttackType.Digital,
    knowledge=Knowledge.Black
)
class BanditPriorAttack(EvasionAttack):
    r"""
    This class implement Bandits and Priors attack from [1]

    The authors draw from the Bandit optimization framework, and try to find a more principled way to estimate gradient
    in a black box threat scenario. The idea is that at each gradient estimation iteration, each estimation is highly
    correlated with the previous one, abrupt changes in the gradient are infrequent,
    therefore exist prior knowledge to be used.
    They frame the problem as an agent whose actions at each round
    are gradient estimation along a certain random direction,
    and the reward is given by -l(v) = -<\nabla L(x_t,y), v_t/|v_t|>,
    where L is the classifier loss, and v_t the current gradient estimation.

    [1] doi.org/10.48550/arXiv.1807.07978
    """
    CONFIG_T = BanditPriorAttackConfig

    def __init__(self, config: EvasionAttackConfig):
        super().__init__(config)
        self._config = cast(BanditPriorAttackConfig, self._config)
        self.iterations = 0
        self.n_queries = 0
        self.unit_ball_proj = LpNormalization(p=self._config.p, radius=1.0, batched=True)
        self.gradient_normalizer = LpNormalization(p=self._config.p, radius=1.0, batched=True)

    @torch.no_grad()
    def gradient_est(self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Gradient estimation of l(v) = -\nabla <\nabla L(x), v>"""
        u = torch.normal(mean=torch.zeros_like(v), std=(1 / v[0].numel()) ** 0.5)  # sqrt bc torch normal require std
        u = self.unit_ball_proj(u)
        # antithetic samples for state and normalize
        q1, q2 = v + self._config.delta * u, v - self._config.delta * u
        q1, q2 = self.unit_ball_proj(q1), self.unit_ball_proj(q2)
        # thanks to simplifications we use the loss of the classifier on the following points
        xq1, xq2 = x + self._config.eps * q1, x + self._config.eps * q2
        pred_q1, pred_q2 = self._config.model(xq1), self._config.model(xq2)
        self.n_queries += 2
        lq1, lq2 = self._config.loss(xq1, x, y, pred_q1), self._config.loss(xq2, x, y, pred_q2)
        grad_est = u * (lq2 - lq1).view(-1, *[1] * (u.dim() - 1)) / (self._config.delta * self._config.eps)
        return grad_est

    def track_variables(self):
        super().track_variables()
        self.add_variable_to_track("perturbation", "images")

    def step(
            self, i: int, x: torch.Tensor, y: Optional[torch.Tensor] = None, ext_results: Optional[dict] = None,
            **kwargs
    ) -> tuple[torch.Tensor, bool]:
        # tqdm and verbose logging
        loop = kwargs.get("loop")
        log_on_tqdm = self._config.verbose and loop is not None and hasattr(loop, "set_postfix")
        x, y = x.float(), y.float()
        # Initialize states
        if not hasattr(self, "x_adv"):
            self.x_adv = x.clone().detach().requires_grad_(True)
            self.x_adv_active = self.x_adv
        if not hasattr(self, "state"):
            self.state = torch.zeros_like(x, device=self._config.device, requires_grad=True)
            self.state_active = self.state
        if not hasattr(self, "y_active"):
            self.y_active = y
        if not hasattr(self, "active_id_list"):
            self.active_id_list = torch.arange(x.shape[0], device=self._config.device)
        if not hasattr(self, "perturbation_optim"):
            self.perturbation_optim = self._config.optimizer([self.x_adv], **self._config.perturbation_optimizer_params)
        if not hasattr(self, "state_optim"):
            self.state_optim = self._config.optimizer([self.state], **self._config.state_optimizer_params)
        if not hasattr(self, "baseline_preds"):
            self.baseline_preds = self._config.model(x).argmax(dim=-1)
        if not hasattr(self, "preds"):
            self.preds = self.baseline_preds.clone()
        if not hasattr(self, "perturbation"):
            self.perturbation = self.x_adv - x

        # 1 - Query model, filter batch for active samples
        # query model
        pred = self._config.model(self.x_adv_active)
        # get mask on active / inactive samples
        misclassified_samples = _compare_misclassification(self.y_active, pred)
        # convert current active samples to ids ranging on original object size
        for idx, misclassified in enumerate(misclassified_samples):
            if misclassified:
                self.preds[self.active_id_list[idx]] = pred[idx].argmax(dim=-1)

        # get mask of original object size
        self.active_id_list = self.active_id_list[
            ~misclassified_samples
        ]  # remove misclassified samples from active list
        active_mask = torch.zeros(self.x_adv.shape[0], dtype=torch.bool, device=self._config.device)
        active_mask[self.active_id_list] = True

        # print(f"Active samples: {self.active_id_list} at iteration {i}, THEN X ADV SHAPE: {self.x_adv_active.shape}")

        if torch.all(misclassified_samples):
            if self._config.verbose:
                print(misclassified_samples)
                print(f"All samples misclassified after {i} iterations.")
            return self.x_adv, True
        else:
            self.x_active = x[active_mask]
            self.x_adv_active = self.x_adv[active_mask]
            self.state_active = self.state[active_mask]
            self.y_active = y[active_mask]
            perturbation_normalizer = LpNormalization(
                p=self._config.p, radius=self._config.epsilon, center=self.x_active, batched=True
            )
        if log_on_tqdm:
            loop.set_postfix(active_ids=self.active_id_list)
        # 1 /

        # update gradient estimator state
        # s_t <- s_{t-1} + n * \Delta_t
        # \Delta_t-1 = \nabla_v(<\nabla L(x_t,y), v_{t-1}>) = \nabla_v l(v)
        state_gradient = self.gradient_est(
            self.x_adv_active, self.y_active, self.state_active
        )  # objective is to maximize l(v)
        self.state_optim.zero_grad()
        self.state_active.backward(gradient=state_gradient)
        self.state_optim.step()

        # update perturbation
        # x_t+1 <- x_t + e * s_t / |s_t|
        with torch.no_grad():
            normalized_state = self.gradient_normalizer(self.state_active)  # - bc want gradient ascent
        self.perturbation_optim.zero_grad()
        self.x_adv_active.backward(gradient=-normalized_state)
        self.perturbation_optim.step()

        # projection of input onto ball of valid adversarial examples
        with torch.no_grad():
            self.x_adv[active_mask] = self.x_active + perturbation_normalizer(self.x_adv[active_mask])
            self.x_adv_active = self.x_adv[active_mask]
            self.perturbation = self.x_adv - x

        return self.x_adv, False

    def reset(self):
        super().reset()
        for atr in [
            "state",
            "x_adv",
            "x_adv_active",
            "y_active",
            "state_optim",
            "perturbation_optim",
            "active_id_list"
        ]:
            if hasattr(self, atr):
                delattr(self, atr)
