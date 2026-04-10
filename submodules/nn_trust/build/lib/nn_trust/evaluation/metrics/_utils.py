from typing import Literal

import torch


def _update_robustness(
        atk_robustness: torch.Tensor,
        robustness: torch.Tensor,
        atk_misclassification: torch.Tensor,
        misclassification: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Update the global state of the function
    """

    # If there are two different values for an input that create a misclassification,
    # the minimum value is retained.
    tmp_bool = torch.logical_and(misclassification, atk_misclassification)
    if torch.any(tmp_bool):
        robustness[tmp_bool] = torch.min(robustness[tmp_bool], atk_robustness[tmp_bool])

    # If a misclassification is created for an input where there was nothing before,
    # the new value is inserted.
    tmp_bool = torch.logical_and(torch.logical_not(misclassification), atk_misclassification)
    if torch.any(tmp_bool):
        robustness[tmp_bool] = atk_robustness[tmp_bool]

    # If no misclassification can be found for an input,
    # the highest value is retained.
    tmp_bool = torch.logical_not(torch.logical_or(misclassification, atk_misclassification))
    if torch.any(tmp_bool):
        robustness[tmp_bool] = torch.min(robustness[tmp_bool], atk_robustness[tmp_bool])

    # So `misclassification` now takes into account old misclassifications and new ones.
    misclassification = torch.logical_or(misclassification, atk_misclassification)

    return misclassification, robustness


def _compute_robustness(
        robustness: torch.Tensor,
        misclassification: torch.Tensor,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        reduction: Literal["mean", "min"] = "mean",
) -> float:
    """Computes the robustness score based on the provided tensors.

    Args:
        robustness (torch.Tensor): Tensor of robustness values.
        misclassification (torch.Tensor): Boolean tensor indicating misclassifications.

    Returns:
        float: The computed robustness score.
    """
    out = torch.tensor(0.0, device=device)
    robustness = torch.tensor(robustness)
    misclassification = torch.tensor(misclassification)

    if torch.any(misclassification):
        if reduction == "mean":
            out += robustness[misclassification].mean()
        elif reduction == "min":
            out += robustness[misclassification].min()

    not_misclassified = torch.logical_not(misclassification)
    if torch.any(not_misclassified):
        out += not_misclassified.float().mean() * robustness[not_misclassified].max()
    return out.item()


def _compute_curvature(
        pred_points: torch.Tensor,
        n_neighbours: int = 10,
        eps: float = 1e-6,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        **kwargs,
) -> float:
    r"""Estimate the average curvature of a manifold given a collection of points on
    the manifold.

    :param points: A list of tensors where each tensor represents a point on the manifold.
    :type points: :class:`List[torch.Tensor]`
    :param neighbors: The number of nearest neighbors to consider for each point, default is 10.
    :type neighbors: *int*
    :param eps: Regularization parameter for numerical stability purposes.
    :type eps: *float*
    :type delta: *float*

    :returns: The estimated average curvature of the manifold as a float and a :class:`torch.Tensor`
        that represents the largest :math:`1-delta`-percentile weighted angle for each data point.
    """

    n_points, d = pred_points.shape
    if n_points <= 1:
        # In this case, the number of points are insufficient to compute the curvature of the space
        out = 1e6

    else:
        n_neighbours = min(n_neighbours, n_points - 1)

        # distances has shape (n_points, n_points)
        distances = torch.cdist(pred_points, pred_points)
        # for each point it is taken the closest n_neighbours points.

        neighbors_idxs = torch.topk(distances, k=n_neighbours + 1)[1][:, 1:]

        USigmas = torch.zeros((n_points, d, n_neighbours), device=device)
        Sigmas = torch.zeros((n_points, n_neighbours), device=device)

        pred_points = pred_points.transpose(-1, -2)
        for i in range(n_points):
            # Get the closest point in L2 norm
            G_i = pred_points[:, neighbors_idxs[i]]
            M_i = G_i - torch.mean(G_i, dim=-1, keepdim=True)
            # Computing the SVD
            U, S, _ = torch.linalg.svd(M_i, full_matrices=False)

            USigmas[i], Sigmas[i] = torch.matmul(U, torch.diag(S)), S

        out = 0.0
        for i in range(n_points):
            # Cache the values stored in the tensors
            USigma_i, Sigma_i, neigh_idxs = USigmas[i], Sigmas[i], neighbors_idxs[i]

            # Generate the Q matrices of the neighbor of the point i
            # 1 x 1 x N_i x d  @  N_i x d x N_i =>  1 x N_1 x N_i x N_i
            # Qs = torch.matmul(USigma_i.T.unsqueeze(0).unsqueeze(0), USigmas[neigh_idxs]).squeeze(0)
            Qs = torch.einsum("ji,bjk->bik", USigma_i, USigmas[neigh_idxs])
            # Get all svd decomposition of the Q matrices
            UQs, SQs, _ = torch.vmap(torch.linalg.svd)(Qs)
            # Computes the division between the traces of the previous Singular value
            # parts in the neighborhood of i and take the average.
            arg_arcos = SQs.sum(dim=-1) / (Sigma_i * Sigmas[neigh_idxs]).sum(dim=-1).clamp_min(eps)

            theta_i = torch.acos(arg_arcos.clamp(-1.0 + eps, 1.0 - eps))
            out += theta_i.mean().item() / n_points
    return out
