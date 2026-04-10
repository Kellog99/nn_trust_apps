from math import ceil
from typing import Optional

import torch
from pydantic import Field

from nn_trust.attack import EvasionAttack, EvasionAttackConfig, EvasionAttackFactory
from nn_trust.attack.utils._utils import (
    _compare_misclassification,
    batched_argmin_line_search,
    boundary_search,
    dct_basis_2d,
)
from nn_trust.core import AttackType, Knowledge, Task


class GeoDAAttackConfig(EvasionAttackConfig):
    budget: int = Field(
        default=50,
        description="Maximum number of queries to run on a given model",
        gt=0
    )
    lmbd: float = Field(
        default=0.55,
        description="Value of the exponential for determining the optimal queries per iteration.",
        gt=0.0
    )


@EvasionAttackFactory.register(
    name="Geometric Decision Based Attack",
    description="A black-box adversarial attack that tries to find the closest boundary point aimed at misclassify the initial input.",
    task={Task.Classification},
    type=AttackType.Digital,
    knowledge=Knowledge.Black
)
class GeoDAAttack(EvasionAttack):
    r"""Implements a black-box adversarial attack based on similar arguments as DeepFool (See [1]_). The core idea is to
    find the closest boundary point. This is achieved via an iterative procedure of randomly sampling directions
    and providing the closest point intersecting the decision boundary hyperplane with respect to the initial
    given sample. Herafter, we give a small documentation of known implementations:

    :class:`GeoDAAttackConfig` provides parameters used for the configuration of the attack that are passed as
    a ``config`` variable in the initialized of :class:`GeoDAAttack`.

    :param config: The configuration can be tweaked via changing the following parameters:
        1. ``p`` which specifies the :math:`L^p`-norm to be used in the normalization procedures,
        2. ``budget`` the maximum number of queries for each sample that can be used.
        3. ``lmbd`` a parameter that should be optimal in range ``[0.4, 0.7]`` representing the convergence rate
            of the optimization procedure (See Discussion of [1]_).
        4. ``max_iters`` the maximum number of iterations of the optimization procedure.

    Example::

    Consider a batch of images :math:`(B, C, H, W)`, which we denote as ``DATA_INPUT``, and a corresponding
    one-hot encoded labels ``TARGET_LABEL`` of shape :math:`(B, N_c)`. Let ``MODEL`` be a :class:`ModelAdapter`.
    Then, we can use the attack as follows

    >>> from nn_trust.attack import EvasionAttackFactory
    >>> cnf = EvasionAttackFactory.get_config('geoda',
    >>>         model=MODEL,
    >>>         task=Task.Classification,
    >>>         targeted=True,
    >>>         budeget=10000,
    >>>         max_iters=100)
    >>> atk = EvasionAttackFactory.create(config=cnf)
    >>> atk.generate(DATA_INPUT, TARGET_LABEL)

    .. [1] Rahmati, Ali, Seyed-Mohsen Moosavi-Dezfooli, Pascal Frossard and Huaiyu Dai.
        “GeoDA: A Geometric Framework for Black-Box Adversarial Attacks.”
        2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2020): 8443-8452.
    """

    CONFIG_T = GeoDAAttackConfig

    def optimal_queries_per_iterations(self) -> list[int]:
        r"""Computes the optimal number of queries per iteration as described in
        formula (19) of the paper. Moreover, as pointed out in Subsection 6.3, we
        use ``70`` as the optimal number of queries for the first iteration of the
        algorithm.
        """
        lmbd = self._config.lmbd
        T = self._config.max_iters
        N = self._config.budget
        max_val = (lmbd ** (-2 / 3 * (T + 1)) - 1) / (lmbd ** (-2 / 3) - 1)

        queries = [70]
        queries.extend([ceil(N * lmbd ** (-2 / 3 * t) / max_val + 1) for t in range(1, T)])
        return queries

    def track_variables(self):
        super().track_variables()
        self.add_variable_to_track("perturbation", "images")
        self.add_variable_to_track("misclassified", "tensor")

    @torch.no_grad()
    def step(
            self,
            i: int,
            x: torch.Tensor,
            y: Optional[torch.Tensor] = None,
            ext_results: Optional[dict] = None,
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
        # Note the indexing is 1.. N+1 in generate
        i = i - 1
        # Initialize the adversary sample
        if not hasattr(self, "x_adv"):
            self.x_adv = x.clone()
        if not hasattr(self, "perturbation"):
            self.perturbation = self.x_adv - x

        # Cache the values of queries per iteration and q.
        if not hasattr(self, "queries_per_iteration"):
            self.queries_per_iteration = self.optimal_queries_per_iterations()
        if not hasattr(self, "q"):
            # Computes q s.t. 1/q + 1/p = 1.
            if self._config.p == 1:
                self.q = float("inf")
            elif self._config.p == float("inf"):
                self.q = 1.0
            else:
                self.q = self._config.p / (self._config.p - 1)

        # Initialize the discrete cosine basis
        if not hasattr(self, "discrete_cosine_basis") or self.discrete_cosine_basis is None:
            self.discrete_cosine_basis = dct_basis_2d(x.shape[-2], device=self._config.device)

        # Early stop
        if hasattr(self, "misclassified"):
            if torch.all(self.misclassified):
                return self.x_adv, True

        # Define constants: view_shape, discrete_cosine_basis and the optimal queries per iterations.
        batch_shape = (-1, *([1] * (self.x_adv.dim() - 1)))

        # Define the evaluator function that is used by binary search and boundary search.
        def evaluator(tmp):
            nonlocal y
            return _compare_misclassification(y[: tmp.shape[0]], tmp, dim=-1)

        omegas_Nt = torch.zeros_like(self.x_adv)
        if self._config.verbose:
            print("Range of omegas at initialization: ", omegas_Nt.view(-1).min(), omegas_Nt.view(-1).max())

        # Computes the estimator of the gradient omega_Nt
        for t in range(self.queries_per_iteration[i]):
            # sample the directions eta_i and computes the rho
            random_sample = torch.randn_like(omegas_Nt) / (omegas_Nt.numel() ** (1 / (self._config.p + 1)))
            # maps via the discrete_cosine_basis
            random_sample = torch.matmul(random_sample, self.discrete_cosine_basis)

            if self._config.verbose:
                print(f"Query to estimate the boundary decision distance thingo: {t}")
                print("random sample range", random_sample.view(-1).min(), random_sample.view(-1).max())
            rhos = _compare_misclassification(y, self._config.model(self.x_adv + random_sample), dim=-1) * 2.0 - 1.0
            vv = rhos.view(*batch_shape) * random_sample
            omegas_Nt.add_(vv)
            if self._config.verbose:
                print(f"{rhos=}")
                print(f"Range of omegas after the {t} estimate: ", omegas_Nt.view(-1).min(), omegas_Nt.view(-1).max())
        # Takes the mean of the previous random direction weighted by their mis-classification result.
        omegas_Nt.div_(self.queries_per_iteration[i])
        # Actually computes the omegas_Nt by normalizing the estimator mu_Nt
        omegas_Nt.div_(
            omegas_Nt.view(omegas_Nt.shape[0], -1)
            .norm(p=self.q, dim=-1)
            .view(*batch_shape)
            .clamp_min(self._config.toll)
        )

        # Perform an expansion on the specified direction of the estimator omega_Nt on that line.
        max_ts = boundary_search(
            model=self._config.model, evaluator=evaluator, x0=self.x_adv, dx=omegas_Nt, step_size=2.0, max_iters=200
        ).clamp_max(1.0)
        if self._config.verbose:
            print("Maximum ts: ", max_ts)
        # Binary search to get the least scalar value \hat{r}_t such that it mis-classifies, i.e. we reduce the
        # ts found in the previous boundary_search step.
        minimal_rs = batched_argmin_line_search(
            model=self._config.model,
            evaluator=evaluator,
            x0=self.x_adv,
            dx=omegas_Nt,
            min_t=self._config.toll,
            max_t=max_ts + self._config.toll,
            max_iters=200,
            max_eps=self._config.toll,
        )

        if self._config.verbose:
            print(f"{minimal_rs=}")

        # Update the estimated minimal direction to exit the classification region.
        omegas_Nt.mul_(minimal_rs.view(*batch_shape))
        self.perturbation += omegas_Nt
        self.x_adv = torch.clamp(self.x_adv + omegas_Nt, min=-1.0, max=1.0)
        # Take the best adv we have and updates those that are still mis-classified
        self.misclassified = _compare_misclassification(y, self._config.model(self.x_adv), dim=-1)

        if torch.all(self.misclassified):
            return self.x_adv, True

        return self.x_adv, False

    def reset(self):
        super().reset()
        for atr in [
            "discrete_cosine_basis",
            "x_adv",
            "q",
            "queries_per_iteration",
            "misclassified",
            "perturbation"
        ]:
            if hasattr(self, atr):
                delattr(self, atr)

    def __repr__(self) -> str:
        return "Geometric Decision Based Attack (GeoDA)"
