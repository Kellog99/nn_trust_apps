import base64
import io
import math
import os
from typing import Literal, get_args, get_origin

from PIL import Image
from annotated_types import Gt, Ge, Le, Lt
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

from attack_server.lib.model import ParametersProps

def b64str_to_pil(b64_image_str: str) -> Image.Image:
    image_bytes = base64.b64decode(b64_image_str)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def pil_to_b64str(pil_image: Image.Image) -> str:
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def _param_name(id: str, info: FieldInfo) -> str:
    return getattr(info, "title") or id


def _parse_bounds(metadata: list) -> tuple[float, float, bool, bool]:
    lo, hi = 0.0, 1000.0
    has_lo, has_hi = False, False
    for val in metadata:
        if isinstance(val, (Gt, Ge)):
            lo = getattr(val, "ge" if isinstance(val, Ge) else "gt")
            has_lo = True
        elif isinstance(val, (Lt, Le)):
            hi = getattr(val, "le" if isinstance(val, Le) else "lt")
            has_hi = True
    return lo, hi, has_lo, has_hi


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def get_parameter_prop(id: str, param_info: FieldInfo) -> ParametersProps:
    name = _param_name(id, param_info)
    ann = param_info.annotation

    # --- enum (Literal or str) ---
    if get_origin(ann) is Literal:
        options = [str(o) for o in get_args(ann)]
        default = param_info.default if param_info.default is not PydanticUndefined else options[0]
        return ParametersProps(id=id, name=name, default=str(default),
                               description=param_info.description, kind="enum", options=options)

    if ann is str:
        default = param_info.default if param_info.default is not PydanticUndefined else ""
        return ParametersProps(id=id, name=name, default=str(default),
                               description=param_info.description)

    # --- number ---
    is_int = ann is int
    lo, hi, has_lo, has_hi = _parse_bounds(param_info.metadata)
    if is_int and not has_lo:
        lo = 1

    if math.isinf(hi) or hi > 1e10:
        hi = 1000
    if math.isinf(lo) or lo < -1e10:
        lo = 0

    default = param_info.default
    if default is PydanticUndefined:
        default = (hi + lo) / 2
    else:
        default = float(default)
        if not has_hi and 0 <= default <= 1:
            hi = 1
        if not has_lo and default < lo:
            lo = min(0, default)
        if not has_hi and default > hi:
            hi = default * 2
        default = _clamp(default, lo, hi)

    if lo > hi:
        lo, hi = hi, lo

    step = getattr(param_info, "step", None)
    if step is None:
        step = (hi - lo) / 10000
        if is_int:
            step = max(int(step), 1)

    return ParametersProps(
        id=id, name=name,
        min=float(lo), max=float(hi), step=float(step),
        default=float(default), description=param_info.description,
    )


# ------------------ JOBS utility --------------------------
def find_image(start_dir: str):
    """
    Depth-first search through directories starting at `start_dir`
    to find the first image file. Once found, return the path
    relative to `start_dir`.
    Directories and files are explored in alphabetical order.
    """
    image_exts = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif', '.svg'}
    stack = [start_dir]
    visited = set()

    while stack:
        path = stack.pop()
        try:
            if os.path.islink(path):
                continue

            if os.path.isdir(path):
                real = os.path.realpath(path)
                if real in visited:
                    continue
                visited.add(real)

                try:
                    entries = list(os.scandir(path))
                except PermissionError:
                    continue

                # Sort entries alphabetically by name
                entries.sort(key=lambda e: e.name.lower(), reverse=True)
                # reverse=True because we’re using a stack (LIFO), so we push reversed order
                for entry in entries:
                    stack.append(entry.path)

            else:
                _, ext = os.path.splitext(path)
                if ext.lower() in image_exts:
                    abs_path = os.path.abspath(path)
                    return os.path.join(start_dir.split(os.sep)[-1], os.path.relpath(abs_path, start_dir))
        except Exception:
            continue

    return None
