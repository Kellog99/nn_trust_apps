from typing import Literal, get_args, get_origin, Any

import torch
from annotated_types import Gt, Ge, Le, Lt
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

from models.model import ParametersProps
from services.utils.image import b64str_to_pil, tensor_image_to_b64str
from torchvision.utils import draw_bounding_boxes
from nn_trust.attack.utils.detection import xywh2xyxy


_DEFAULT_LO, _DEFAULT_HI = 0.0, 200.0


def _param_name(id: str, info: FieldInfo) -> str:
    return getattr(info, "title", None) or id


def _get_value(value: Any, default: Any) -> Any:
    return default if value is PydanticUndefined else value


def _parse_bounds(metadata: list[Any]) -> tuple[float, float]:
    lo, hi = _DEFAULT_LO, _DEFAULT_HI
    for val in metadata:
        if isinstance(val, (Gt, Ge)):
            lo = float(getattr(val, "ge" if isinstance(val, Ge) else "gt"))
        elif isinstance(val, (Lt, Le)):
            hi = float(getattr(val, "le" if isinstance(val, Le) else "lt"))
    return lo, hi


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def get_parameter_prop(
        id: str,
        param_info: FieldInfo,
        max_value: int = 200,
) -> ParametersProps:
    """Characterizes an attack's parameter as a ParametersProps object."""
    name = _param_name(id, param_info)
    ann = param_info.annotation

    if get_origin(ann) is Literal:
        options = [str(o) for o in get_args(ann)]
        default = str(_get_value(param_info.default, options[0]))
        return ParametersProps(
            id=id, name=name, default=default,
            description=param_info.description, kind="enum", options=options,
        )

    if ann is str:
        default = str(_get_value(param_info.default, ""))
        return ParametersProps(
            id=id, name=name, default=default,
            description=param_info.description, kind="string",
        )

    if ann is bool:
        default = bool(_get_value(param_info.default, False))
        return ParametersProps(
            id=id, name=name, default=default,
            description=param_info.description, kind="boolean",
        )

    is_int = ann is int
    lo, hi = _parse_bounds(param_info.metadata)
    lo = max(lo, 0.0)
    hi = min(hi, float(max_value))

    raw_default = _get_value(param_info.default, None)
    # Zero is a valid and meaningful default for several optimizer
    # parameters (for example FOM's momentum and dampening).  Do not use
    # truthiness here, otherwise an explicit default of 0 is replaced by the
    # midpoint of the allowed range.
    default = (lo + (hi - lo) / 2) if raw_default is None else raw_default
    default = _clamp(default, lo, hi)

    if lo >= hi:
        raise ValueError(f"For the parameter {id}, the min ({lo!r}) must be strictly less than max ({hi!r})")

    step = getattr(param_info, "step", None)
    if step is None:
        step = (hi - lo) / max_value
        if is_int:
            step = max(round(step), 1)

    return ParametersProps(
        id=id,
        name=name,
        min=lo,
        max=hi,
        step=float(step),
        default=default,
        description=param_info.description,
    )

def filter_predictions(pred, display_top_k):
    """
    Filter predictions based on top_k
    """
    boxes = pred["boxes"].detach().cpu()
    labels = pred["labels"].detach().cpu()
    scores = pred["scores"].detach().cpu() 

    idx = torch.arange(len(labels))

    # rank the predictions based on scores and select top_k
    if display_top_k is not None and idx.numel() > display_top_k:
        idx = idx[scores[idx].topk(display_top_k).indices]

    return {
        "boxes": boxes[idx],
        "labels": labels[idx],
        "scores": scores[idx],
    }


def draw_predictions(image, pred, display_top_k, class_names=None):
    """
    Draw predictions on the image
    """

    # filter predictions based on top_k
    pred = filter_predictions(pred, display_top_k)

    # convert image to uint8 and get its height and width
    image_uint8 = (image.detach().cpu().clamp(0, 1) * 255).to(torch.uint8)
    _, h, w = image_uint8.shape


    boxes = pred["boxes"]

    # convert boxes from xywh to xyxy format
    boxes = xywh2xyxy(boxes)

    # convert boxes to absolute coordinates if they are in relative coordinates
    if boxes.numel() > 0 and boxes.max() <= 1.5:
        boxes[:, [0, 2]] *= w
        boxes[:, [1, 3]] *= h

    labels_tensor = pred["labels"]

    labels = [
        str(class_names[int(label)]) if class_names is not None else str(int(label))
        for label in labels_tensor
    ]

    # if scores are available, format the labels with their corresponding scores
    if "scores" in pred:
        labels = [
            f"{name}: {float(score):.2f}"
            for name, score in zip(labels, pred["scores"])
        ]

    return draw_bounding_boxes(
        image_uint8,
        boxes,
        labels=labels,
        width=2,
        colors="red",
    )
