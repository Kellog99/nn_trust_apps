import importlib
from importlib import util
from pathlib import Path
from typing import Optional

import timm
import torch

from nn_trust import ModelAdapter, Task
from nn_trust.models.api_model import APIModel


def load_model(
        model_type: str,
        model_path: Optional[str | Path] = None,
        model_api: Optional[str] = None,
        model_id: Optional[str] = None,
        task: Task = Task.Classification,
        **kwargs
) -> ModelAdapter:
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

    model_path : str or pathlib.Path, optional
        Path to the required model file or directory, depending on `model_type`.

    model_id : str, optional
        Reserved for future extensions (e.g., loading from model registries).

    model_api : str, optional
        Base URL for remote inference when `model_type="api"`.

    task: Task
        Task that the model has to perform
    Returns
    -------
    ModelAdapter
        A unified adapter wrapping the loaded model.
    """
    if isinstance(model_path, str):
        model_path = Path(model_path).expanduser()

    model = None
    match model_type:
        case "plain":
            # This is the case where a plain model is given in a pth file
            # hence it is sufficient to load the pth
            if model_path is None:
                raise ValueError("The model's class does not exist in the python file.")
            model_checkpoint_path = model_path / "model.pth"
            assert model_checkpoint_path.exists() and model_checkpoint_path.is_file()
            model = torch.load(
                model_checkpoint_path,
                weights_only=False,
                map_location=torch.device('cpu')
            )
        case "timm":
            model = timm.create_model(
                model_name=model_id,
                pretrained=True,
            )
        case "model_weights":
            model_checkpoint_path = model_path / "model_state_dict.pth"
            model_definition_path = model_path / "model.py"
            assert model_checkpoint_path.exists() and model_definition_path.exists()
            spec = importlib.util.spec_from_file_location("model", str(model_definition_path))
            model_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(model_module)
            model = model_module.Model()
            model.load_state_dict(
                torch.load(
                    model_checkpoint_path,
                    weights_only=True,
                    map_location=torch.device('cpu')
                )
            )
        case "torch_script":
            model_checkpoint_path = model_path / "model.pth"
            model = torch.jit.load(
                model_checkpoint_path,
                map_location=torch.device('cpu')
            )
        case "torch_dynamo":
            model_checkpoint_path = model_path / "model.pth"
            model = torch.export.load(model_checkpoint_path)
        case "api":
            if model_api is None:
                raise ValueError("The model's is not given.")
            model = APIModel(
                api_url=model_api,
                task=task
            )

    if model is None:
        raise ValueError(
            f"The model type, {model_type}, does not match with the available types."
        )

    return ModelAdapter(
        model=model,
        task=task,
    )
