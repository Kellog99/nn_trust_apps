from typing import Literal, Tuple
import pathlib
import importlib
import json

import torch

from nn_trust.core import ModelAdapter, Task
from .onnx_model import ONNXModel
from .ultralytics_models import UltralyticsModel
from .api_model import APIModel
from .hf_model import HFModel
from pydantic import BaseModel, field_validator, model_validator

class ModelLoadConfig(BaseModel):
    type: Literal["plain", "model_weights", "torch_script", "torch_dynamo", "onnx", "api", "huggingface"]
    file_path: str | pathlib.Path | None = None
    api_url: str | None = None
    name: str
    num_classes: int
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    task: str = "classification"
    input_size: int | Tuple[int, int]

    @field_validator('task')
    @classmethod
    def validate_task(cls, v) -> Task:
        """Validate task set is not empty."""
        if isinstance(v, str):
            v = Task.from_str(v)
        return v

    @field_validator('file_path')
    @classmethod
    def validate_path(cls, v) -> pathlib.Path:
        """Validate task set is not empty."""
        if isinstance(v, str):
            v = pathlib.Path(v)
        return v

    @model_validator(mode="after")
    def check_model_or_api(self):
        """
        Require exactly one of:
        - model_path is not None
        - api_url is not None
        """
        if (self.file_path is None) and (self.api_url is None) and (self.type != "huggingface"):
            raise ValueError("Either model_path or api_url must be provided. When not using remote repos like HF.")

        if (self.file_path is not None) and (self.api_url is not None):
            raise ValueError("Provide only one: model_path or api_url, not both.")

        return self



def load_model(
        model_path: str | pathlib.Path,
        model_config: dict | ModelLoadConfig | None = None,
        **kwargs
):
    """
    Load a model and wrap into the relative `ModelAdapter` interface
    starting multiple serialization formats.

    Parameters
    ----------
    model_type : {"plain", "model_weights", "torch_script", "torch_dynamo", "onnx", "api"}
        Selects the loading method:
        - **"plain"**: Load a raw Python model object saved with `torch.save(...)`.
          `file_path` must point to a `.pth` or similar file containing a pickled model.

        - **"model_weights"**: Load a model defined by a Python module plus a
          `state_dict`. `file_path` must be a directory containing:
              - `model.py` defining a class `Model`
              - `model_state_dict.pth` containing weights

        - **"torch_script"**: Load a TorchScript model created with `torch.jit.trace`
          or `torch.jit.script`. `file_path` must point to a valid `.pt` file.

        - **"torch_dynamo"**: Load a model exported with `torch.export`.
          `file_path` (.pt2) must contain an exported model that can be loaded with
          `torch.export.load`.

        - **"onnx"**: Load an ONNX model using `ONNXModel`. `file_path` must point
          to a `.onnx` file. The model is loaded on the specified `device`.

        - **"api"**: Load a remote model accessed through a custom HTTP API.
          Requires `model_api_url`.

    file_path : str or pathlib.Path, optional
        Path to the required model file or directory, depending on `model_type`.

    model_id : str, optional
        Reserved for future extensions (e.g., loading from model registries).

    model_api_url : str, optional
        Base URL for remote inference when `model_type="api"`.

    device : str
        Device used for ONNX execution; ignored by other backends.

    Returns
    -------
    ModelAdapter
        A unified adapter wrapping the loaded model.
    """
    model_info = {}
    if model_path is not None:
        model_path = pathlib.Path(model_path)
        path_is_dir = model_path.is_dir()
        path_has_info_file = (model_path / "info.json").is_file()
        if path_is_dir and path_has_info_file:
            with open(model_path / "info.json", "r") as model_info_file:
                model_info = json.load(model_info_file)
        else:
            raise FileNotFoundError(f"Folder {model_path} is not a directory.")
        model_info["file_path"] = str(model_path)
    elif kwargs.get("type", "") == "huggingface":
        model_info["file_path"] = None
    else:
        raise ValueError("Model path is required if not using remote repos")
    kwargs = {k:v for k,v in kwargs.items() if v is not None}
    model_info = model_info | kwargs

    model_info = ModelLoadConfig(**model_info)
    device = torch.device(model_info.device)
    match model_info.type:
        case "plain":
            model_checkpoint_path = model_path / "model.pth"
            assert model_checkpoint_path.exists() and model_checkpoint_path.is_file()
            model = torch.load(model_checkpoint_path, weights_only=False, map_location=torch.device('cpu'))
            model = ModelAdapter(model, task=model_info.task)
            model = model.to(device)
        case "model_weights":
            model_checkpoint_path = model_path / "model_state_dict.pth"
            model_definition_path = model_path / "model.py"
            assert model_checkpoint_path.exists() and model_definition_path.exists()
            spec = importlib.util.spec_from_file_location("model", str(model_definition_path))
            model_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(model_module)
            model = model_module.Model()
            model.load_state_dict(torch.load(model_checkpoint_path, weights_only=True, map_location=torch.device('cpu')))
            model = ModelAdapter(model, task=model_info.task)
            model = model.to(device)
        case "torch_script":
            model_checkpoint_path = model_path / "model.pth"
            model = torch.jit.load(model_checkpoint_path, map_location=torch.device('cpu'))
            model = ModelAdapter(model, task=model_info.task)
            model = model.to(device)
        case "torch_dynamo":
            model_checkpoint_path = model_path / "model.pth"
            model = torch.export.load(model_checkpoint_path)
            model = ModelAdapter(model.module(), task=model_info.task)
            model = model.to(device)
        case "onnx":
            model_checkpoint_path = model_path / "model.onnx"
            model = ONNXModel(model_checkpoint_path, device=model_info.device, task=model_info.task)
        case "api":
            model = APIModel(api_url=model_info.api_url, task=model_info.task)
        case "huggingface":
            if model_path is not None:
                model_checkpoint_path = model_path / "model_state_dict.pth"
                if not model_checkpoint_path.exists():
                    model_checkpoint_path = None
            else:
                model_checkpoint_path = None
            model = HFModel(model_name=model_info.name,
                            checkpoint_path=model_checkpoint_path,
                            device=model_info.device,
                            num_labels=model_info.num_classes,
                            task=model_info.task
            )
            model = model.to(device)
        case _:
            raise ValueError(f"Unsupported mode: {model_info.type}")
    model.input_size = model_info.input_size
    model.num_classes = model_info.num_classes
    try:
        model.eval()
    except Exception as e:
        print(f"Impossibile to set model to eval mode, for model type {model_info.type}: {e}")
    return model
