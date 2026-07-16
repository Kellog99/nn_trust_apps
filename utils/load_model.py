import json
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
import timm
import torch

from models import ModelInfo
from nn_trust import CVModelAdapter, Task
from nn_trust.models import HFCVModel, ONNXCVModel, APICVModel


# ---------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------
def _require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_model(
        model_path: str | Path,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        **kwargs
):
    r"""
    Load a model and wrap into the relative `ModelAdapter` interface
    starting multiple serialization formats.

    Args:
        model_path
        device

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

    Returns
    -------
    CVModelAdapter
        A unified adapter wrapping the loaded model.
    """

    if model_path is not None:
        model_path = Path(model_path)
        path_is_dir = model_path.is_dir()
        path_has_info_file = (model_path / "info.json").is_file()
        if path_is_dir and path_has_info_file:
            with open(model_path / "info.json", "r") as model_info_file:
                model_info = json.load(model_info_file)
        else:
            raise FileNotFoundError(f"Folder {model_path} is not a directory.")
        model_info["file_path"] = str(model_path)
    else:
        raise ValueError("Model path is required if not using remote repos")

    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    model_info = model_info | kwargs

    model_info = ModelInfo(**model_info)
    device = torch.device(device)
    task = Task.from_str(model_info.task)

    match model_info.model_type:
        case "plain":
            model_checkpoint_path = model_path / "model.pth"
            assert model_checkpoint_path.exists() and model_checkpoint_path.is_file()
            model = torch.load(
                f=model_checkpoint_path,
                weights_only=False
            )
            model = CVModelAdapter(model, task=task)
        case "timm":
            model = timm.create_model(
                model_name=model_info.id,
                pretrained=True,
            )
        case "model_weights":
            model_checkpoint_path = model_path / "model_state_dict.pth"
            model_definition_path = model_path / "model.py"
            assert model_checkpoint_path.exists() and model_definition_path.exists()
            spec = spec_from_file_location("model", str(model_definition_path))
            model_module = module_from_spec(spec)
            spec.loader.exec_module(model_module)
            model = model_module.Model()
            model.load_state_dict(
                torch.load(
                    f=model_checkpoint_path,
                    weights_only=True
                )
            )
            model = CVModelAdapter(model, task=task)

        case "torch_script":
            model_checkpoint_path = model_path / "model.pth"
            model = torch.jit.load(model_checkpoint_path, map_location=torch.device('cpu'))
            model = CVModelAdapter(model, task=task)

        case "torch_dynamo":
            model_checkpoint_path = model_path / "model.pth"
            model = torch.export.load(model_checkpoint_path)
            model = CVModelAdapter(model.module(), task=task)

        case "onnx":
            model_checkpoint_path = model_path / "model.onnx"
            model = ONNXCVModel(
                model_filepath=model_checkpoint_path,
                device=device,
                task=task
            )

        case "api":
            model = APICVModel(
                api_url=model_info.api_url,
                task=task
            )

        case "huggingface":
            if model_path is not None:
                model_checkpoint_path = model_path / "model_state_dict.pth"
                if not model_checkpoint_path.exists():
                    model_checkpoint_path = None
            else:
                model_checkpoint_path = None
            model = HFCVModel(model_name=model_info.name,
                              checkpoint_path=model_checkpoint_path,
                              device=device,
                              num_labels=model_info.num_classes,
                              task=task
                              )
            model = model
        case _:
            raise ValueError(f"Unsupported mode: {model_info.model_type}")
    model.num_classes = model_info.num_classes
    model = model.to(device)
    model.eval()
    return model
