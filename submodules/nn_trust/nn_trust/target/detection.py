from typing import Literal

import torch
import torch.nn as nn

from .functional import (
    mirror_bounding_box,
    target_diversion_topleft_corner_detection,
)


class MirrorBoundingBoxTarget(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, bounding_boxes: list[torch.Tensor]) -> torch.Tensor:
        return mirror_bounding_box(bounding_boxes)


class ShrinkTopLeftCornerBoundingBoxTarget(nn.Module):
    def __init__(self, in_fmt: str = "cxcywh", delta: float = 0.1):
        super().__init__()
        if delta < 0.0 or delta > 1.0:
            raise ValueError(f"The value 'delta' must be larger than 0 and less than 1, given value {delta}")
        if in_fmt in ['cxcywh', 'xyxy', 'xywh']:
            raise ValueError(
                f"The value 'in_fmt' must be any of the following strings: 'cxcywh', 'xyxy', 'xywh'. However '{in_fmt}' was given.")
        self._delta = delta
        self._in_fmt = in_fmt

    def forward(self, y: list[torch.Tensor]) -> torch.Tensor:
        return target_diversion_topleft_corner_detection(y, delta=self._delta, in_fmt=self._in_fmt)
