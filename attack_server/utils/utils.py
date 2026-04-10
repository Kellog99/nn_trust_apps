import base64
import io
import math
import os

from PIL import Image
from annotated_types import Gt, Ge, Le, Lt
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

from attack_server.lib.model import ParametersProps


######################### CONVERSION #########################
def b64str_to_pil(b64_image_str: str) -> Image.Image:
    """
    from a base64 encoded string to a PIL image
    """
    image_bytes = base64.b64decode(b64_image_str)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return image


def pil_to_b64str(pil_image: Image.Image) -> str:
    """
    from a PIL image to a base64 encoded string
    """
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    adv_img_base64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return adv_img_base64_str


##############################################################


def get_parameter_prop(
        id: str,
        param_info: FieldInfo
) -> ParametersProps:
    """
    This function allows to properly produce a P
    """
    max_value = 1000
    min_value = 1

    if len(param_info.metadata) > 0:
        # Extracting from the metadata the maximum value and minimum value of the parameters
        # If there are no constraints than the max value and the min value are the one above indicated
        for val in param_info.metadata:
            if isinstance(val, (Gt, Ge)):
                min_value = getattr(val, 'ge' if isinstance(val, Ge) else 'gt')
            elif isinstance(val, (Lt, Le)):
                max_value = getattr(val, 'le' if isinstance(val, Le) else 'lt')

    # Handle infinity values - replace with reasonable defaults
    if math.isinf(max_value) or max_value > 1e10:
        max_value = 1000
    if math.isinf(min_value) or min_value < -1e10:
        min_value = 0 if param_info.annotation == float else 1

    # Ensure min < max
    if min_value > max_value:
        tmp = min_value
        min_value = max_value
        max_value = tmp

    # The default value, if not assigned, is the mean of the interval
    if param_info.default is PydanticUndefined:
        default = (max_value + min_value) / 2
    else:
        default = param_info.default
        # Clamp default to valid range
        default = max(min_value, min(max_value, default))

    if hasattr(param_info, 'step'):
        step = getattr(param_info, 'step')
    else:
        step = (max_value - min_value) / 10000
        if id == "lr":
            step = 1e-6
            max_value = 1
            min_value = 1e-3
        if isinstance(param_info.annotation, int) or param_info.annotation == int:
            step = max(int(step), 1)

    name = getattr(param_info, "title") if hasattr(param_info, "title") and getattr(param_info, "title") != None else id
    return ParametersProps(
        id=id,
        name=name,
        min=float(min_value),
        max=float(max_value),
        step=float(step),
        default=float(default),
        description=param_info.description
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
