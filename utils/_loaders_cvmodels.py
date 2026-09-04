from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path

import timm
import torch

from models import ModelInfo
from nn_trust import CVModelAdapter, Task
from nn_trust.models import HFCVModel, ONNXCVModel, APICVModel
from nn_trust.models.ultralytics_models import UltralyticsCVModel


def _require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Expected file at {path}, but it does not exist.")
    return path


def _load_plain(
        model_path: Path,
        task: Task,
        **args
) -> CVModelAdapter:
    ckpt = _require_file(model_path / "model.pth")
    model = torch.load(f=ckpt, weights_only=False)
    return CVModelAdapter(model, task=task)


def _load_timm(
        model_id: str,
        task: Task,
        **args
) -> CVModelAdapter:
    if model_id is None:
        raise ValueError("It is necessary to provide the model's id for loading it from 'timm' library.")
    model = timm.create_model(
        model_name=model_id,
        pretrained=True
    )
    return CVModelAdapter(model, task=task)


def _load_model_weights(
        model_path: Path,
        task: Task,
        **args
) -> CVModelAdapter:
    ckpt = _require_file(model_path / "model_state_dict.pth")
    definition = _require_file(model_path / "model.py")

    spec = spec_from_file_location("model", str(definition))
    model_module = module_from_spec(spec)
    spec.loader.exec_module(model_module)

    model = model_module.Model()
    model.load_state_dict(torch.load(f=ckpt, weights_only=True))
    return CVModelAdapter(model, task=task)


def _load_torch_script(
        model_path: Path,
        task: Task,
        **args
) -> CVModelAdapter:
    ckpt = _require_file(model_path / "model.pt")
    model = torch.jit.load(ckpt, map_location=torch.device("cpu"))
    return CVModelAdapter(model, task=task)


def _load_torch_dynamo(
        model_path: Path,
        task: Task,
        **args
) -> CVModelAdapter:
    ckpt = _require_file(model_path / "model.pt2")
    exported = torch.export.load(ckpt)
    return CVModelAdapter(exported.module(), task=task)


def _load_onnx(
        model_path: Path,
        task: Task,
        **args) -> CVModelAdapter:
    ckpt = _require_file(model_path / "model.onnx")
    return ONNXCVModel(model_filepath=ckpt, task=task)


def _load_api(
        api_url: str,
        task: Task,
        **args
) -> CVModelAdapter:
    if not api_url:
        raise ValueError("model_info.api_url is required for the 'api' model type.")
    return APICVModel(
        api_url=api_url,
        task=task
    )


def _load_huggingface_cv(
        model_path: Path,
        info: ModelInfo,
        task: Task,
        **args
) -> CVModelAdapter:
    ckpt = model_path / "model_state_dict.pth"
    checkpoint_path = ckpt if ckpt.exists() else None
    return HFCVModel(
        model_name=info.name,
        checkpoint_path=checkpoint_path,
        num_labels=info.num_classes,
        task=task,
    )

def _load_ultralytics(
        model_id: str | None,
        model_path: Path,
        task: Task,
        **args,
    ):
    if task != Task.Detection:
        raise ValueError("The 'ultralytics' loader only supports detection models.")

    ckpt = model_path / "model.pt"
    model_name = str(ckpt) if ckpt.is_file() else model_id

    if model_name is None:
        raise ValueError("The 'ultralytics' loader requires either model.pt or a model id.")

    return UltralyticsCVModel(model_name=model_name)