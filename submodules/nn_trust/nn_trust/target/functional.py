from typing import Literal

import torch

def target_misclassification_onehot(
        x: list[int] | list[torch.Tensor],
        num_classes: int | None = None
    ) -> torch.Tensor:
    """Creates a misclassification target from a ''ground truth'' tensor.

    The result of this transformation is a one-hot encoding tensor

    :param x: a ground truth tensor of shape :math:`(B, K_1, \dots, K_r)`.
    :param num_classes: number of classes for the one-hot encoding vector.

    :returns: a one-hot encoding vector :class:`torch.Tensor` of shape :math:`(B, K_1, \dots, K_r, N_\text{cls})`.
    """
    if not isinstance(x, list):
        raise ValueError("The input is not a list")
    if num_classes is None:
        num_classes = torch.max(x.view(-1)).int().item()
    x = torch.stack(x) if isinstance(x[0], torch.Tensor) else torch.tensor(x)
    return torch.nn.functional.one_hot(x, num_classes=num_classes).float()


def target_avoid_misclassification_onehot(
        x: list[int] | list[torch.Tensor],
        num_classes: int | None = None
    ) -> torch.Tensor:
    """Creates a one-hot encoding tensor with negative value on the passed class `x`.

    :param x: list of indexes or tensors representing the class to avoid.
    :param num_classes: total number of classes.

    :returns: A :class:`torch.Tensor` one-hot encoding the selected class with a negative value.
    """
    if not isinstance(x, list):
        raise ValueError("The input is not a list.")

    if not x:
        raise ValueError("The list is empty.")

    if num_classes is None:
        num_classes = torch.max(x.view(-1)).int().item()
    x = torch.stack(x) if isinstance(x[0], torch.Tensor) else torch.tensor(x)
    return -torch.nn.functional.one_hot(x, num_classes=num_classes).float()

def target_exclude_one_label_misclassification_onehot(
        x: list[int] | list[torch.Tensor],
        num_classes: int,
        reduction: Literal["none", "mean"] = "mean"
    ):
    """Given a list of hard-label list, it generates an objective target in fuzzy one-hot encoding
    which is :math:`0` on the selected labels and positive on the other labels.

    :param num_classes: Number of classes to perform the one-hot encoding.
    :param reduction: Applies a reduction on the result. If ``'none'``, no reduction is applied the result should
        be a one-hot encoding with values either 1 or 0. If ``'mean'`` the positive values are scaled by the number
        of non-zero values, hence the result for each batch element is a discrete probability distribution.
        Default is ``'mean'``.

    Example::

    >>> obj_target = OnehotExcludeOneLabelTarget(num_classes=5)
    >>> obj_target(t_classes=[0, 1])
    tensor([[0.0000, 0.1250, 0.1250, 0.1250, 0.1250], [0.1250, 0.0000, 0.1250, 0.1250, 0.1250]])
    """
    evade_target = torch.ones((1, num_classes))
    all_one_hot_to_evade = torch.nn.functional.one_hot(torch.tensor(x), num_classes=num_classes)
    evade_target = torch.vmap(lambda x: evade_target - x)(all_one_hot_to_evade).squeeze(1)
    if reduction == "mean":
        evade_target /= evade_target.count_nonzero()
    return evade_target

def target_diversion_topleft_corner_detection(
        x: list[torch.Tensor] | torch.Tensor,
        delta: float,
        in_fmt: str
    ) -> torch.Tensor:
    """Creates a detection for each detection bounding box passed that has IoU less than `delta` with the detection bounding box mask.

    :param x: a list of :class:`torch.Tensor` representing the bounding box of a detection result with shape :math:`(D, 4)` or a single
    :param delta: between `x` and :meth:`target_diversion_randomly_sample_segmentation(x, delta)` we should have an IoU
        that is less than or equal to `delta`.
    :param in_fmt: a string represeting the format of the bounding boxes. It can be ``'xyxy'`` or ``'xywh'`` or ``'cxcywh'``.

    :returns: A target bounding box of same shape of the stacked ``x`` tensors.
    """

    if isinstance(x, list):
        x = torch.stack(x)  # B, D, H, W with D representing the number of detection/mask per batch

    B, D, _ = x.shape
    top_left_corner = torch.zeros((B, D, 2), dtype=torch.float32)
    if in_fmt == "xyxy":
        top_left_corner = x[..., -2:]
    elif in_fmt == "xywh":
        top_left_corner = x[..., :2]
    elif in_fmt == "cxcywh":
        top_left_corner = x[..., :2] - x[..., -2:] / 2

    width_height = torch.zeros((B, D, 2), dtype=torch.float32)
    if in_fmt == "xyxy":
        width_height = x[..., 2:] - x[..., :2]
    elif in_fmt == "xywh" or in_fmt == "cxcywh":
        width_height = x[..., -2:]

    scaling = torch.ones_like(width_height) * (delta ** 0.5)
    bottom_right_corner = top_left_corner + (width_height) * scaling
    targets = torch.zeros_like(x)
    new_width_height = bottom_right_corner - top_left_corner
    if in_fmt == "xyxy":
        targets = torch.cat([top_left_corner, bottom_right_corner], dim=-1)
    elif in_fmt == "xywh":
        targets = torch.cat([top_left_corner, new_width_height], dim=-1)
    elif in_fmt == "cxcywh":
        targets = torch.cat([top_left_corner / 2 + bottom_right_corner / 2, new_width_height], dim=-1)
    return targets


def mirror_bounding_box(x: list[torch.Tensor] | torch.Tensor) -> torch.Tensor:
    """Mirrors all bounding boxes under the assumption that the bounding boxes have value
    in range [0, 1].

    :param x: a tensor of bounding boxes or a list of bounding boxes.

    :returns: A tensor of bounding boxes mirrored on the x and y axis.
    """
    if isinstance(x, list):
        x = torch.stack(x)  # B, D, H, W with D representing the number of detection/mask per batch
    return 1. - x