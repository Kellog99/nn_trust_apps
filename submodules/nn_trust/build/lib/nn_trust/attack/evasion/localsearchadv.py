from typing import Optional

import torch
from pydantic import Field

from nn_trust import AttackType, Knowledge, ModelAdapter, Task
from nn_trust.attack import EvasionAttack, EvasionAttackConfig, EvasionAttackFactory


def _cyclic_scaling(x: torch.Tensor, scaling: float, lb: float, ub: float) -> torch.Tensor:
    r"""Scales each coordinate by a ``scaling`` factor if such value is comprised
    between the ``lb`` and the ``ub``.

    :param x: a :class:`torch.Tensor` such that all its values are comprised in the range ``[lb, ub]``.
    :param scaling: a positive floating point.
    :param lb: lower bound.
    :param ub: upper bound.

    :returns: The original tensor scaled by ``scaling`` and rescaled such that
        :math:`x' \in [lb, ub]`.
    """
    if ub < lb:
        raise ValueError("The lower bound must be smaller than the upper bound")

    if torch.logical_or(torch.any(x > ub), torch.any(x < lb)):
        raise ValueError("The input tensor must be consistent with the provided lower and upper bound")

    if scaling < 0:
        raise ValueError("The scaling must be a positive value")

    rang = ub - lb
    scaled_x = x * scaling  # * torch.rand( (x.size(1), ), device=x.device).repeat((x.size(0), 1))
    outsiders = (scaled_x < lb) | (scaled_x > ub)
    scaled_x[outsiders] = scaled_x[outsiders] - rang * ((scaled_x[outsiders] - lb) / rang).floor()

    return scaled_x


def _update_neighborhood(idx: torch.Tensor, d: int, max_d: int, interleave: bool) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Given a set of indexes returns the neighborhood from ``-d`` to ``+d`` for each specified index, excluded the
    original value.

    :param idx: a batch of indexes of shape ``(B, N)``

    :returns: a batch of indexes ``(B, N * 2d)``
    """
    device = idx.device
    mn = 2 * d + 1
    bsize = idx.size(0)
    ksize = idx.size(1)
    if interleave:
        idx = idx.repeat_interleave(mn * mn).view(idx.size(0), -1)
    else:
        idx = idx.repeat(1, mn * mn)
    rang = torch.arange(-d, d + 1, device=device)
    rang = rang.repeat(bsize, mn * ksize)
    idx = idx - rang
    mask = torch.logical_and(idx < max_d, idx >= 0)
    return idx, mask


def _sparse_sign_perturbation(
        x: torch.Tensor,
        p: float,
        idx_x: torch.Tensor,
        idx_y: torch.Tensor,
        best_k_scores: int,
        model: ModelAdapter,
        labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Given a batch of images ``x`` and a batch of indexes on the ``x`` and ``y`` axis,
    at the specified positions we substitute ``x[b, :, p_x, p_y] = p * sign(x[b, :, p_x, p_y])``.
    We use a top-k selection to determine which position should be optimal according the change at the
    specified positions.

    :param x:
    :param p:
    :param idx_x:
    :param idx_y:
    :param model:
    :param labels:

    :returns: the set of best performing adversarial images and the corresponding indexes for each element of the batch.
    """
    if x.dim() < 4:
        raise ValueError("The shape of the input vector 'x' must be on 4 axis")

    if idx_x.dim() == 1:
        idx_x = idx_x.unsqueeze(0)
    if idx_y.dim() == 1:
        idx_y = idx_y.unsqueeze(0)

    if x.size(0) != idx_x.size(0) or x.size(0) != idx_y.size(0):
        raise ValueError("Not enough indexes were provided to select on the batch dimension")
    device = x.device
    leaderboard = torch.ones((x.size(0), best_k_scores), device=device)
    best_x = torch.zeros((x.size(0), best_k_scores), dtype=int, device=device)
    best_y = torch.zeros((x.size(0), best_k_scores), dtype=int, device=device)
    batch_indexing_tensor = torch.arange(x.size(0), device=device)
    for i, (idx, idy) in enumerate(zip(idx_x.unbind(1), idx_y.unbind(1), strict=False)):
        tmp_adv = x.clone()
        tmp_adv[batch_indexing_tensor, ..., idx, idy] = x[batch_indexing_tensor, ..., idx, idy].sign() * p
        # print(f"{batch_indexing_tensor=}", f"{labels=}")
        scores = model(tmp_adv).softmax(dim=-1)[batch_indexing_tensor, labels]
        # Update the leaderboard, use the leaderboard changes to determine which score is actually improved!
        leaderboard, _ = torch.cat([leaderboard, scores.unsqueeze(1)], dim=1).sort(dim=1, descending=False)
        leaderboard = leaderboard[:, :best_k_scores]
        # Finally update the new optimal idxes and images
        equal_idxes = torch.argwhere(leaderboard == scores.repeat(best_k_scores, 1).t())
        ib, js = equal_idxes.unbind(-1)
        best_x[ib, js] = idx[ib]
        best_y[ib, js] = idy[ib]

    return best_x, best_y


def _update_search_parameter(val: float, iteration_n: int, max_iteration: int, delta_loss: float) -> float:
    """Updates the learning rate parameter ``val``."""
    if iteration_n > max_iteration / 2:
        val *= 0.6
    elif iteration_n > max_iteration // 10:
        val *= 0.99
    return val


class LocalSearchAttackConfig(EvasionAttackConfig):
    scaling: float = Field(
        default=43,
        description="Scaling factor used to find the criticality of each pixel.",
        ge=0.0
    )
    epsilon: float = Field(
        default=1.1,
        description="Scaling of the perturbation function.",
        ge=0.0,
        le=2.0
    )
    positions: float = Field(
        default=0.05,
        description="Percentage of points to select.",
        ge=0.0,
        le=1.0
    )
    neighborhood: int = Field(
        default=2,
        description="Neighborhood points to check.",
        ge=0)
    k_misclassification: int = Field(
        default=1,
        description="Top-k misclassification loss to use.",
        gt=0
    )
    top_k_images: int = Field(
        default=5,
        description="Top-k perturbations to pass to the next round of optimization.",
        gt=0
    )
    cooldown: int = Field(
        default=30,
        description="Number of iterations to wait for a point to be updated again.",
        ge=0
    )
    pre_cooldown: int = Field(
        default=0,
        description="Number of iterations a point might be before starting the cooldown.",
        ge=0
    )

@EvasionAttackFactory.register(
    name="Local Search Attack",
    description="A black-box adversarial attack that tries to find the most critical pixels in the image and modify them.",
    task={Task.Classification},
    type=AttackType.Digital,
    knowledge=Knowledge.White
)
class LocalSearchAttack(EvasionAttack):
    r"""Implementation of Algorithm 3 of [1]_.

    The idea is to find critical points by searching on the image positions and computing the score of the classifier
    with respect to a modified image in a single point. Then, if such position is 'critical', we perform a modification
    by cycling and rescaling such pixel value.

    :param max_iters: number of iterations to run the search algorithm, corresponds to the parameter :math:`R` in Algorithm 3 of [1]_.
    :param scaling: the scaling used for the cyclic perturbation of a given pixel. It determines the pixel's __criticality__. It corresponds to the parameter :math:`r` of [1]_.
    :param perturbation: the scaling used to change the value of a given pixel for each channel, i.e. parameter :math:`p` of Algorithm 3 of [1]_.
    :param positions: it is the percentage of points used in the initial warmup stage of the searching algorithm specified in [1]_.
    :param neighborhood: a positive integer used to determine how large the search space should be around an initial random point.
    :param k_misclassification: in [1]_ they use the top-k mis-classification. This parameter determines how many k we should use
        to consider an image to be mis-classified.
    :param top_k_images: a positive integer counting the maximum number of different images
    :param image_range_lb: the lower bound value for each component of the image.
    :param image_range_ub: the upper bound value for each component of the image.

    .. [1] Narodytska, Nina and Shiva Prasad Kasiviswanathan. “Simple Black-Box Adversarial Perturbations for Deep Networks.” ArXiv abs/1612.06299 (2016)
    """

    CONFIG_T = LocalSearchAttackConfig
    def track_variables(self):
        super().track_variables()
        self.add_variable_to_track("adv_x", "images")
        self.add_variable_to_track("perturbation", "images")

    @torch.no_grad()
    def step(
            self,
            i: int,
            x: torch.Tensor,
            y: Optional[torch.Tensor] = None,
            ext_results: Optional[dict] = None,
            **kwargs
    ) -> tuple[torch.Tensor, bool]:
        r"""Follows Algorithm 3 of [1]_. The idea is to find critical points using the sign of the image at a given pixel
        position, then slightly modify those pixel that are considered __critical__.

        :param x: A batch of images with shape ``(B, C, H, W)``, respectively the batch size, number of channels, height and width.
        :param y: The labels of the corresponding images ``x``. They could be either in one-hot encoding or a vector of integers.

        .. [1] Narodytska, Nina and Shiva Prasad Kasiviswanathan. “Simple Black-Box Adversarial Perturbations for Deep Networks.” ArXiv abs/1612.06299 (2016)
        """
        # tqdm and verbose logging
        loop = kwargs.get("loop")
        log_on_tqdm = self._config.verbose and loop is not None and hasattr(loop, "set_postfix")

        # Initialize points 'direction'
        batch_size, _, height, width = x.shape
        if not hasattr(self, "points_positions_x"):
            points_x = int(width * self._config.positions)
            self.points_positions_x = torch.randint(0, width, (batch_size, points_x), device=self._config.device)
        if not hasattr(self, "points_positions_y"):
            points_y = int(height * self._config.positions)
            self.points_positions_y = torch.randint(0, height, (batch_size, points_y), device=self._config.device)
        # Create a blacklist of points to not modify for a specific amount of time.
        # The blacklist is indexed by a tuple (b, x, y) : T with b being the element in the batch, x, y
        # the position and T the time to 'cooldown'.
        if not hasattr(self, "points_blacklist"):
            self.points_blacklist = {}

        # Create the adversarial attack
        if not hasattr(self, "adv_x"):
            self.adv_x = x.clone()
        if not hasattr(self, "perturbation"):
            self.perturbation = self.adv_x - x
        if not hasattr(self, "tmp_adv_to_attack"):
            self.tmp_adv_to_attack = x.clone()
        if not hasattr(self, "still_to_attack"):
            self.still_to_attack = torch.ones((batch_size,), dtype=bool, device=self._config.device)

        if not hasattr(self, "initial_avg_loss"):
            self.initial_avg_loss = None
        if not hasattr(self, "prev_loss"):
            self.prev_loss = None
        # Perturbation parameter
        pp = self._config.epsilon

        # Generate some perturbed images and select only the few 'optimal'
        # Note: if we can skip the generation of those indices, whenever an adversarial perturbation
        # image has been found in the batch, it would reduce greatly the inference time.
        best_idxs_x, best_idxs_y = _sparse_sign_perturbation(
            self.adv_x,
            self._config.scaling,
            self.points_positions_x,
            self.points_positions_y,
            best_k_scores=self._config.top_k_images,
            model=self._config.model,
            labels=y.argmax(dim=-1),
        )
        # The result of adversarial_xs is a collection of images with shape (best_k_scores, batch_size, channels, height, width)
        # best_idxs_x is of shape (batch_size, best_k_scores) and similarly for best_idxs_y.
        # Now take the top scores use the cycling re-scaling
        for j, (opx, opy) in enumerate(zip(best_idxs_x.unbind(0), best_idxs_y.unbind(0), strict=False)):
            self.tmp_adv_to_attack[j, :, opy, opx] = _cyclic_scaling(
                self.adv_x[j, :, opy, opx], scaling=pp, lb=-1.0, ub=1.0
            )
            # The idea is to know whether the topk predictions are corresponding to the passed label,
            # if not, misclassification occurred, hence we can skip it.
            probs = self._config.model(self.tmp_adv_to_attack).softmax(dim=-1)
            if self.prev_loss is None:
                self.prev_loss = probs[:, y.argmax(dim=-1)].mean().item()

            if self.initial_avg_loss is None:
                self.initial_avg_loss = probs[:, y.argmax(dim=-1)].mean().item()

            # Determine which image to update
            top_k_predictions = torch.topk(probs, k=self._config.k_misclassification, dim=-1).indices
            still_to_attack_new = torch.any(top_k_predictions == y.argmax(dim=-1).unsqueeze(-1), dim=-1)
            update = torch.logical_or(torch.logical_not(still_to_attack_new), self.still_to_attack)
            self.adv_x[update] = self.tmp_adv_to_attack[update]
            still_to_attack = torch.logical_and(self.still_to_attack, still_to_attack_new)
            # Update the search parameter pp based on the logit average score
            pp = _update_search_parameter(
                pp, i, self._config.max_iters, probs[:, y.argmax(dim=-1)].mean().item() - self.prev_loss
            )

            if log_on_tqdm:
                loop.set_postfix(
                    {
                        "imagesLeft": still_to_attack.sum().item(),
                        "absAvgDiffs": (x - self.adv_x).view(x.size(0), -1).norm(dim=-1).mean().item(),
                        "pointsEvaluated": self.points_positions_x.size(1),
                        "avgScore": probs[:, y.argmax(dim=-1)].mean().item(),
                        "initialAvgScore": self.initial_avg_loss,
                        "pp": pp,
                    }
                )

            if self.still_to_attack.sum() < 1:
                self.tmp_adv_to_attack = self.adv_x.clone()
                self.prev_loss = probs[:, y.argmax(dim=-1)].mean().item()
                self.perturbation = self.adv_x - x
                return self.adv_x, True
            # Updates the neighborhood of points to check at the next iteration
            self.points_positions_x, mask_x = _update_neighborhood(
                best_idxs_x, self._config.neighborhood, width, interleave=False
            )
            self.points_positions_y, mask_y = _update_neighborhood(
                best_idxs_y, self._config.neighborhood, height, interleave=True
            )
            # Take the mask between x and y, then
            # Reduce the size to be consistent between batches.
            mm = torch.logical_and(mask_x, mask_y)
            top_kk_mm = mm.sum(dim=-1).min().item()
            mm = torch.logical_and(mm, mm.cumsum(dim=-1) <= top_kk_mm)
            self.points_positions_x = self.points_positions_x[mm].view(batch_size, -1)
            self.points_positions_y = self.points_positions_y[mm].view(batch_size, -1)
            # Select the points using the blacklist
            # NOTE: we could iterate on the blacklist instead, but we rather choose to iterate on the number of
            # searched points because it is constant, while the blacklist is increasing in size at each iteration.
            for b in range(self.points_positions_x.size(0)):
                for k in range(self.points_positions_x.size(1)):
                    m = (b, self.points_positions_x[b, k].item(), self.points_positions_y[b, k].item())
                    if m in self.points_blacklist and self.points_blacklist[m] >= -self._config.pre_cooldown:
                        self.points_positions_x[b, k] = -1
                        self.points_positions_y[b, k] = -1
                    elif m in self.points_blacklist and self.points_blacklist[m] < -self._config.pre_cooldown:
                        self.points_blacklist[m] = self._config.cooldown
                    elif m not in self.points_blacklist:
                        self.points_blacklist[m] = 0

            # Prune the set of indices by selecting as few valid points in a batch
            mask = self.points_positions_x >= 0
            top_kk = mask.sum(dim=-1).min().item()
            new_mask = mask.cumsum(dim=-1) <= top_kk
            mask = torch.logical_and(mask, new_mask)
            self.points_positions_x = self.points_positions_x[mask].view(
                batch_size, -1
            )  # torch.randint(0, width, points_positions_x.size(), device=self._config.device)[~mask]
            self.points_positions_y = self.points_positions_y[mask].view(
                batch_size, -1
            )  # torch.randint(0, height, points_positions_y.size(), device=self._config.device)[~mask]

            # Update blacklist
            keys = list(self.points_blacklist.keys())
            for k in keys:
                if self.points_blacklist[k] > 0:
                    self.points_blacklist[k] = self.points_blacklist[k] - 1
                else:
                    _ = self.points_blacklist.pop(k)
            self.tmp_adv_to_attack = self.adv_x.clone()
            self.prev_loss = probs[:, y.argmax(dim=-1)].mean().item()
            self.perturbation = self.adv_x - x
            return self.adv_x, False

    def reset(self):
        super().reset()
        for atr in [
            "tmp_adv_to_attack",
            "still_to_attack",
            "adv_x",
            "points_positions_x",
            "points_positions_y",
            "points_blacklist",
            "prev_loss",
            "initial_avg_loss",
            "perturbation"
        ]:
            if hasattr(self, atr):
                delattr(self, atr)

    def __repr__(self):
        return "Local Search Adversarial Attack"
