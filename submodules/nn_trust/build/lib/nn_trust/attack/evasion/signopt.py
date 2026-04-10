from typing import Annotated, Optional, cast

import torch
from annotated_types import Ge, Le
from pydantic import Field

from nn_trust.attack import EvasionAttack, EvasionAttackConfig, EvasionAttackFactory
from nn_trust.attack.utils._utils import _binary_line_search, _binary_line_search_geometric, _compare_misclassification
from nn_trust.core import AttackType, Knowledge, Task


class SignOPTAttackConfig(EvasionAttackConfig):
    epsilon: float = Field(
        default=0.1,
        description="Scaling for local derivatives.",
        gt=0.0,
        title="Derivatives' scaling factor"
    )

    max_line_search_queries: int = Field(
        default=80,
        description="Number of line-search optimization steps.",
        gt=0,
        title="Line-search steps."
    )
    initial_directions: int = Field(
        default=100,
        description="Initial number of directions for the attack.",
        gt=0,
        title="Initial directions"
    )
    error: float = Field(
        default=0.01,
        description="Maximum allowed error before stopping.",
        ge=0.0,
        lt=1.0,
        title="Maximum number of errors"
    )
    Q: int = Field(
        default=100,
        description="Number of samples to estimate optimal gradient.",
        gt=0,
        title="Number of samples"
    )
    lr: float = Field(
        default=0.1,
        description="Learning rate for optimization procedure.",
        gt=0.0,
        lt=10.0,
        title="Learning rate"
    )


@EvasionAttackFactory.register(
    name="SignOPT",
    description="A black-box attack that tries to estimate the gradient using solely the model's hard-label information by randomly sampling directions.",
    task={Task.Classification},
    type=AttackType.Digital,
    knowledge=Knowledge.White
)
class SignOPTAttack(EvasionAttack):
    """
    SignOPT: an attack for the hard-label black-box attack setting to generate adversarial examples,
    where limited model queries are allowed and only the decision is provided to a queried data input.
    Sign-OPT attack consistently requires 5× to 10× fewer queries than OPT.[1]

    General description:
    The goal is to efficiently estimate the gradient using solely the model's hard-label information by
    randomly sampling directions. This estimate is then combined with a binary search approach (see [2])
    to enhance the overall method.

    [1] Minhao Cheng, Simranjit Singh, Patrick Chen, Pin-Yu Chen, Sijia Liu, Cho-Jui Hsieh,
        "Sign-OPT: A Query Efficient Hard-Label Adversarial Attack",
        https://doi.org/10.48550/arXiv.1909.10773
    [2] Minhao Cheng, Thong Le, Pin-Yu Chen, Huan Zhang, JinFeng Yi, and Cho-Jui Hsieh.
        "Query-efficient hard-label black-box attack: An optimization-based approach",
        https://openreview.net/forum?id=rJlk6iRqKX.
    """

    CONFIG_T = SignOPTAttackConfig

    def __init__(self, config: EvasionAttackConfig):
        super().__init__(config)
        self._config = cast(SignOPTAttackConfig, self._config)

    def track_variables(self):
        super().track_variables()
        self.add_variable_to_track("g_thetas", "tensor")
        self.add_variable_to_track("thetas", "images")
        self.add_variable_to_track("perturbation", "images")

    @torch.no_grad()
    def step(
            self, i: int, x: torch.Tensor, y: Optional[torch.Tensor] = None, ext_results: Optional[dict] = None,
            **kwargs
    ) -> tuple[torch.Tensor, bool]:
        if not hasattr(self, "x_adv"):
            self.x_adv = x.clone()
        if not hasattr(self, "perturbation"):
            self.perturbation = self.x_adv - x

        if self._config.verbose:
            import time

            prev_time = time.time()
        # Check whether it is the initial step or not
        if i <= 1:
            self.g_thetas = torch.zeros(x.size(0), device=x.device)
            self.thetas = torch.zeros_like(x)
            for b_idx in range(x.size(0)):
                # Sample random directions, then if it happens to send the initial data to a
                # mis-classification region, store the direction.
                guess_directions = torch.randn(
                    (self._config.initial_directions, *x.shape[1:]), device=self._config.device
                )
                self.thetas[b_idx], g_theta = torch.randn_like(x[b_idx], device=self._config.device), float("inf")
                vals = _compare_misclassification(
                    y[b_idx].repeat(self._config.initial_directions, *y.shape[1:]),
                    self._config.model(x[b_idx] + guess_directions),
                    dim=-1,
                )
                # check if we have mis-classification, if not retry
                retry = 0
                while (sum(vals) < 1) and retry < 10:
                    guess_directions = torch.randn(
                        (self._config.initial_directions, *x.shape[1:]), device=self._config.device
                    )
                    self.thetas[b_idx], g_theta = torch.randn_like(x[b_idx], device=self._config.device), float("inf")
                    vals = _compare_misclassification(
                        y[b_idx].repeat(self._config.initial_directions, *y.shape[1:]),
                        self._config.model(x[b_idx] + guess_directions),
                        dim=-1,
                    )
                    retry += 1
                mis_dirs = guess_directions[vals]
                # Computes the scaling of the directions and the directions.
                mis_dirs_lambda = torch.norm(mis_dirs, p=self._config.p, dim=list(range(len(mis_dirs.shape))[1:]))
                mis_dirs /= mis_dirs_lambda.reshape((-1, *([1] * len(mis_dirs.shape[1:]))))
                # Finds the minimal lambda value between 0 and the given value on the given direction.
                # If the lambda value is the minimum, store it as the best mis-classification direction theta.
                for j in range(mis_dirs.size(0)):
                    if self._config.verbose:
                        print(f"Iterating through the initial guesses: {j} of {mis_dirs.shape[0]}.")

                    def obj_func(dt):
                        return _compare_misclassification(
                            y[b_idx].unsqueeze(0), self._config.model(x[b_idx] + dt * mis_dirs[j].unsqueeze(0)), dim=-1
                        ).item()

                    lbd = _binary_line_search(
                        x_min=1e-6,
                        x_max=mis_dirs_lambda[j].item(),
                        objective_func=obj_func,
                        max_iters=self._config.max_line_search_queries,
                        epsilon=self._config.error,
                        verbose=self._config.verbose,
                    )
                    if lbd < g_theta:
                        self.g_thetas[b_idx] = lbd
                        self.thetas[b_idx] = mis_dirs[j]

                ## In case no g_theta has been found, exit the process with a RuntimeError.
                # if g_theta >= float('inf'):
                # warnings.warn(f"No initial direction has been found.", RuntimeWarning)
                # self.x_adv = x + self.g_thetas.view(-1, *([1] * (x.dim() -1))) * self.thetas
                # return self.g_thetas.view(-1, *([1] * (x.dim() -1))) * self.thetas

                if self._config.verbose:
                    print(
                        f"The line search method found: {self.g_thetas[b_idx]},\n"
                        f"The best direction has L^2 norm: {torch.norm(self.thetas[b_idx], p=2.0)}"
                    )
        else:
            for b_idx in range(x.size(0)):
                # Sample random directions to estimate the gradient into we are heading to change theta.
                u_dir = torch.randn((self._config.Q, *x.shape[1:]), device=self._config.device)
                normalized_theta_dirs = self.thetas[b_idx] + self._config.epsilon * u_dir
                # Normalize in L2 each of the Q directions.
                normalized_theta_dirs /= torch.norm(
                    normalized_theta_dirs,
                    p=self._config.p,
                    # normalize wrt all dirs except the batch direction!
                    dim=list(range(len(normalized_theta_dirs.shape))[1:]),
                ).view((-1, *([1] * len(normalized_theta_dirs.shape[1:]))))

                # Fast computation of sign(g(theta + eps * u_i) - g(theta))
                sign_g_new_dir = (
                        torch.logical_not(
                            _compare_misclassification(
                                y[b_idx].repeat(self._config.Q, *y.shape[1:]),
                                self._config.model(x[b_idx] + self.g_thetas[b_idx] * normalized_theta_dirs),
                                dim=-1,
                            )
                        )
                        * 1.0
                )
                sign_g_new_dir.sub_(0.5)
                sign_g_new_dir.mul_(2.0)
                # Computation of the average of the sign * dirs
                # About the view(-1, ...): we need to have the same number of axis of u_dir, however we want to scale
                # each batch by the corresponding sign_g_new_dir element, the other axis are only for shape constraints
                # of parallelization of tensor's component-by-component multiplication.
                g_hat = torch.mean(sign_g_new_dir.view(-1, *([1] * len(u_dir.shape[1:]))) * u_dir, dim=0)
                # Eventually, update theta and compute g_theta.
                self.thetas[b_idx] -= self._config.lr * g_hat
                self.thetas[b_idx] /= torch.norm(self.thetas[b_idx], p=self._config.p)

                def obj_func(dt):
                    return _compare_misclassification(
                        y[b_idx].unsqueeze(0),
                        self._config.model(x[b_idx] + dt * self.thetas[b_idx].unsqueeze(0)),
                        dim=-1,
                    ).item()

                self.g_thetas[b_idx] = _binary_line_search_geometric(
                    self.g_thetas[b_idx] + 1e-6,
                    alpha=0.01,
                    objective_func=obj_func,
                    max_iters=self._config.max_line_search_queries,
                    epsilon=self._config.error,
                    verbose=self._config.verbose,
                )
                if self._config.verbose:
                    print(
                        f"Norm of the modification at iteration {i}: {g_hat.abs().max()}\nNorm of the perturbation: {torch.norm(self.g_thetas.view(-1, *([1] * (x.dim() - 1))) * self.thetas)}."
                    )
                    print(f"Elapsed time per iteration: {time.time() - prev_time}")
                    prev_time = time.time()

        self.perturbation = self.g_thetas.view(-1, *([1] * (x.dim() - 1))) * self.thetas
        self.x_adv = x + self.perturbation
        return self.x_adv, False

    def reset(self):
        super().reset()
        for atr in [
            "x_adv",
            "perturbation",
            "g_thetas",
            "thetas"
        ]:
            if hasattr(self, atr):
                delattr(self, atr)
