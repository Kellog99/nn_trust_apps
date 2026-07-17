from pathlib import Path
from typing import Callable, Optional

import torch

from models.info import MODEL_TYPES
from nn_trust import CVModelAdapter, Task
from utils._loaders import (
    _load_plain,
    _load_api,
    _load_onnx,
    _load_timm,
    _load_huggingface,
    _load_model_weights,
    _load_torch_dynamo,
    _load_torch_script
)

_LOADERS: dict[str, Callable[..., CVModelAdapter]] = {
    "plain": _load_plain,
    "timm": _load_timm,
    "model_weights": _load_model_weights,
    "torch_script": _load_torch_script,
    "torch_dynamo": _load_torch_dynamo,
    "onnx": _load_onnx,
    "api": _load_api,
    "huggingface": _load_huggingface,
}


# ---------------------------------------------------------------------
def load_model(
        model_type: MODEL_TYPES = "plain",
        model_id: Optional[str] = None,
        model_path: Optional[str | Path] = None,
        api_url: Optional[str] = None,
        task: Optional[Task] = None,
        num_classes: Optional[int] = None,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        **kwargs,
):
    r"""
    Load a model from disk and wrap it into the shared `CVModelAdapter`
    interface, supporting multiple serialization formats.

    Args:
        model_type: type of model to load.
        model_path: Directory containing the eventual model to load.
        api_url:
        task
        num_classes:
        device: Target device. Defaults to CUDA if available, else CPU.
        **kwargs: Overrides merged into the fields loaded from
            `info.json` (e.g. `num_classes=`, `task=`).

    Possible `model_type` to use:
        - **"plain"**: raw pickled model via `torch.save`. Expects `model.pth`.
        - **"timm"**: loaded via `timm.create_model(info.id, pretrained=True)`.
        - **"model_weights"**: `model.py` (defining class `Model`) +
          `model_state_dict.pth`.
        - **"torch_script"**: TorchScript module. Expects `model.pt`.
        - **"torch_dynamo"**: `torch.export`-ed module. Expects `model.pt2`.
        - **"onnx"**: ONNX model. Expects `model.onnx`.
        - **"api"**: remote model behind an HTTP API. Requires `info.api_url`.
        - **"huggingface"**: HF model by name, optionally with a local
          `model_state_dict.pth` checkpoint.

    Returns:
        CVModelAdapter: A unified adapter wrapping the loaded model.
    """
    if model_path is None:
        raise ValueError("model_path is required if not using remote repos.")

    try:
        loader = _LOADERS[model_type]
    except KeyError:
        raise ValueError(
            f"Unsupported model type: {model_type}. "
            f"Supported types: {sorted(_LOADERS.keys())}"
        )
    if isinstance(model_path, str):
        model_path: Path = Path(model_path).expanduser().resolve()

    model: CVModelAdapter = loader(
        model_id=model_id,
        model_path=model_path,
        task=task,
        api_url=api_url,
        device=device
    )

    if num_classes:
        model.num_classes = num_classes
    model = model.to(device)
    model.eval()
    return model
