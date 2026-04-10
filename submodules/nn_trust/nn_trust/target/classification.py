from typing import Literal

import torch
import torch.nn as nn

from .functional import (
    mirror_bounding_box,
    target_avoid_misclassification_onehot,
    target_diversion_topleft_corner_detection,
    target_exclude_one_label_misclassification_onehot,
    target_misclassification_onehot,
)


class OnehotTarget(nn.Module):
    r"""Given a list of hard-label list, it generates an objective target in fuzzy one-hot encoding
    which is :math:`1` on the selected label and :math:`0` on the other labels.

    :param num_classes: number of classes of the one-hot encoded vector.

    Example::

    >>> obj_target = OnehotTarget(num_classes=5)
    >>> obj_target([0, 1])
    tensor([[1., 0., 0., 0., 0.], [0., 1., 0., 0., 0.]])
    """

    def __init__(self, num_classes: int):
        super().__init__()
        # TODO: add validation of params?
        self._num_classes = num_classes

    def forward(self, t_classes: list[int]) -> torch.Tensor:
        return target_misclassification_onehot(t_classes, num_classes=self._num_classes)


class AvoidOnehotTarget(nn.Module):
    r"""Given a list of hard-label list, it generates an objective target in fuzzy one-hot encoding
    which is :math:`1` on the selected label and :math:`0` on the other labels.

    :param num_classes: number of classes of the one-hot encoded vector.

    Example::

    >>> obj_target = AvoidOnehotTarget(num_classes=5)
    >>> obj_target([0, 1])
    tensor([[-1., -0., -0., -0., -0.], [-0., -1., -0., -0., -0.]])
    """

    def __init__(self, num_classes: int):
        super().__init__()
        # TODO: add validation of params?
        self._num_classes = num_classes

    def forward(self, t_classes: list[int]) -> torch.Tensor:
        return target_avoid_misclassification_onehot(t_classes, num_classes=self._num_classes)


class OnehotExcludeOneLabelTarget(nn.Module):
    r"""Given a list of hard-label list, it generates an objective target in fuzzy one-hot encoding
    which is :math:`0` on the selected labels and positive on the other labels.

    :param num_classes: Number of classes to perform the one-hot encoding.
    :param reduction: Applies a reduction on the result. If ``'none'``, no reduction is applied the result should
        be a one-hot encoding with values either 1 or 0. If ``'mean'`` the positive values are scaled by the number
        of non-zero values, hence the result for each batch element is a discrete probability distributionn.
        Default is ``'mean'``.

    Example::

    >>> obj_target = OnehotExcludeOneLabelTarget(num_classes=5)
    >>> obj_target(t_classes=[0, 1])
    tensor([[0.0000, 0.1250, 0.1250, 0.1250, 0.1250], [0.1250, 0.0000, 0.1250, 0.1250, 0.1250]])
    """

    def __init__(self, num_classes: int, reduction: Literal["none", "mean"] = "mean"):
        super().__init__()
        # TODO: add validation of params?
        self._num_classes = num_classes
        self._reduction = reduction

    def forward(self, t_classes: list[int]) -> torch.Tensor:
        r"""
        :param t_classes: A list of hard-label that is less than ``self.num_classes``.

        :returns: a float :class:`torch.Tensor`.

        """
        return target_exclude_one_label_misclassification_onehot(
            t_classes, num_classes=self._num_classes, reduction=self._reduction
        )
