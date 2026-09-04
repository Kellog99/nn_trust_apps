import base64
import io
from typing import Literal, get_args, get_origin, Any

import torch
from PIL import Image
from annotated_types import Gt, Ge, Le, Lt
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined
from torchvision.transforms import v2 as T

from models.model import ParametersProps


def b64str_to_pil(b64_image_str: str) -> Image.Image:
    image_bytes = base64.b64decode(b64_image_str)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def tensor_image_to_b64str(image: torch.Tensor) -> str:
    """
    Convert an image tensor to a PNG encoded as a Base64 string.

    Expected shape:
        - (C, H, W), or
        - (1, C, H, W)
    """
    if image.ndim == 4:
        if image.shape[0] != 1:
            raise ValueError(
                f"Expected a batch of size 1, got shape {tuple(image.shape)}"
            )
        image = image[0]

    if image.ndim != 3:
        raise ValueError(
            f"Expected shape (C, H, W), got {tuple(image.shape)}"
        )

    # ToPILImage converts floating-point values to uint8 internally.  Attack
    # outputs can temporarily contain NaN/Inf or values outside the image
    # range, which otherwise produces RuntimeWarnings during that cast.
    image = torch.nan_to_num(image.detach().cpu(), nan=0.0, posinf=1.0, neginf=0.0)
    image = image.clamp(0.0, 1.0)

    pil_img = T.ToPILImage()(image)

    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")

    return base64.b64encode(buffered.getvalue()).decode("utf-8")


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
        return ParametersProps(id=id, name=name, default=default, description=param_info.description)

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
