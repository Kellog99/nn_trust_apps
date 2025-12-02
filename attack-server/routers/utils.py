import math
import os
from pathlib import Path

import PIL
from annotated_types import Gt, Ge, Le, Lt
from nn_trust.attack import EvasionAttackConfig
from nn_trust.attack.attack_factory import EvasionAttackFactory as EAF, AttackInfo
from nn_trust.core import Task
from nn_trust.evaluation.statistic_factory import StatisticsFactory as SF
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined
import logging
import json
from lib.model import ParametersProps


def get_parameter_prop(id: str, param_info: FieldInfo) -> ParametersProps:
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
        step = (max_value - min_value) / 1000
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

def check_model_and_dataset_in_running_jobs(job, d, m):
    """
    Checks if the specified dataset and model to perform a benchmark on are already running.
    """
    return job.get('dataset') == d and job.get('model') == m


def check_attack_already_launched(attacks, req_attacks):
    """
    Checks if one or more specified attacks is already running in the backend.
    """
    return any(item in attacks for item in req_attacks), set(attacks) & set(req_attacks)


def has_aggregate(task_dir: str, dataset: str) -> bool:
    """
    Return True if the given task_dir/<dataset> contains any 'aggregate.json' file.
    """
    task_path = Path(task_dir).expanduser().resolve()
    dataset_dir = task_path / dataset
    if not dataset_dir.is_dir():
        return False
    return any(p.name == "aggregate.json" for p in dataset_dir.rglob("aggregate.json"))


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


def load_models_metadata_from_repo():
    """Load models metadata from repository with structure: model_repo/model_name/{data, info.json}"""
    models = []
    models_root_dir = os.environ.get("INTERNAL_MODEL_STORAGE")
    
    for model_name in os.listdir(models_root_dir):
        model_dir = os.path.join(models_root_dir, model_name)
        # Skip if not a directory
        if not os.path.isdir(model_dir):
            continue
        
        json_path = os.path.join(model_dir, "info.json")
        
        if not os.path.isfile(json_path):
            raise FileNotFoundError(
                f"Missing info.json for model: {model_name}"
            )
        
        # Load model info
        with open(json_path, 'r') as f:
            model_info = json.load(f)
        
        # Determine type based on data file content or extension if needed
        model_type = model_info.get("type", "timm")
        model_info.pop("name")
        model_info.pop("type")
        # Merge info with base entry
        model_entry = {
            "name": model_name,
            "type": model_type,
            **model_info
        }
        
        models.append(model_entry)
        logging.info(f"Loaded model: {model_name}")
    
    return models
# logic related to single attack example
import base64
from typing import Tuple
import PIL
import io
import time
import torch
from torchvision.transforms import v2 as T
from nn_trust.core import ModelAdapter, Task
from nn_trust.target import AvoidOnehotTarget
from torchmetrics.image import StructuralSimilarityIndexMeasure
from nn_trust.models.model_utils import load_model


def b64str_to_pilimage(b64_image_str: str) -> PIL.Image:
    image_bytes = base64.b64decode(b64_image_str)
    image = PIL.Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return image

def pilimage_to_b64str(pilimage: PIL.Image) -> str:
    buffered = io.BytesIO()
    pilimage.save(buffered, format="PNG")
    adv_img_base64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return adv_img_base64_str

def execute_single_image_attack(
        model: ModelAdapter,
        model_input_size: int | Tuple[int, int],
        img: PIL.Image,
        num_classes: int,
        attack_id: str,
        attack_params: dict,
        device: str | None = None,
    ):
    device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(model_input_size, int):
        model_input_size = (model_input_size, model_input_size)

    transformations = T.Compose([
        T.Resize(model_input_size),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
    ])

    x = transformations(img).to(device).unsqueeze(0)
    labels = model(x).argmax(-1).tolist()
    target = AvoidOnehotTarget(num_classes=num_classes)(labels)

    atk_cnf = EAF.get_config(
        class_id=attack_id,
        model=model,
        task=Task.Classification,
        device=device,
        **attack_params
    )
    atk = EAF.create(
        class_id=attack_id,
        config=atk_cnf
    )
    start = time.time()
    x_adv = atk.generate(x=x, y=target).detach()
    end = time.time()

    pert = x_adv - x
    y_adv = model(x_adv).argmax(-1)

    x_adv_pil = T.ToPILImage()(x_adv.squeeze())
    pert_pil = T.ToPILImage()(pert.squeeze())

    ssim_metric = StructuralSimilarityIndexMeasure().to(device)
    return {
        "x": img,
        "y": str(labels[0]),
        "x_adv": x_adv_pil,
        "y_adv": str(y_adv.argmax(-1).item()),
        "pert": pert_pil,
        "metrics":{
            "ssim": ssim_metric(x, x_adv).item(),
            "distance": torch.norm(pert, p=1).item(),
            "execution_time": end - start,
        }
    }
