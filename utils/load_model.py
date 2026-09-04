from pathlib import Path
from typing import Callable, Optional

import torch

from models.info import MODEL_TYPES, ModelInfo
from nn_trust import CVModelAdapter, Task, Knowledge, NLPModelAdapter
from utils._loader_nlp_models import _load_ollama, _load_huggingface_nlp
from utils._loaders_cvmodels import (
    _load_plain,
    _load_api,
    _load_onnx,
    _load_timm,
    _load_huggingface_cv,
    _load_model_weights,
    _load_torch_dynamo,
    _load_torch_script,
)


def load_huggingface_model(
        task: Task,
        model_path: Optional[Path] = None,
        model_id: Optional[str] = None,
        info: Optional[ModelInfo] = None,
        knowledge: Optional[Knowledge] = None,
        **kwars,
) -> CVModelAdapter | NLPModelAdapter:
    """
    It has to switch the loading between the CV and the NLP model
    """
    match task:
        case Task.Classification:
            if model_path is None:
                raise ValueError("model_path is required for Classification")
            if info is None:
                raise ValueError("info is required for Classification")

            return _load_huggingface_cv(
                model_path=model_path,
                info=info,
                task=task,
                **kwars,
            )
        case Task.Classification:
            if model_id is None:
                raise ValueError("model_id is required for getting the Hugging face model.")
            if knowledge is None:
                raise ValueError("knowledge is required for getting the Hugging face model.")
            return _load_huggingface_nlp(
                model_id=model_id,
                knowledge=knowledge,
                task=task,
                **kwars,
            )
        case _:
            raise ValueError(f"Unsupported task: {task}")


_LOADERS: dict[MODEL_TYPES, Callable[..., CVModelAdapter | NLPModelAdapter]] = {
    "Ollama": _load_ollama,
    "HuggingFace": load_huggingface_model,
    "plain": _load_plain,
    "timm": _load_timm,
    "model_weights": _load_model_weights,
    "torch_script": _load_torch_script,
    "torch_dynamo": _load_torch_dynamo,
    "onnx": _load_onnx,
    "api": _load_api,
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
    interface, supporting multiple serialization formats. LLM models are
    wrapped into the `NLPModelAdapter` subclasses `HuggingFaceNLPAdapter`
    (HuggingFace causal LM) or `OllamaNLPAdapter` (remote Ollama API).

    Args:
        model_id
        model_type: type of model to load.
        model_path: Directory containing the eventual model to load.
        api_url:
        task
        num_classes:
        device: Target device. Defaults to CUDA if available, else CPU.
        **kwargs: Overrides merged into the fields loaded from
            `info.json` (e.g. `num_classes=`, `task=`).

    Returns:
        CVModelAdapter | NLPModelAdapter: A unified adapter wrapping the
        loaded model.
    """

    # HuggingFace can be a CV model (local checkpoint via the CV path below)
    # or an LLM (hub causal LM). Ambiguity is resolved by task: a Language
    # task selects the HuggingFaceNLPAdapter; anything else falls through to
    # the unchanged CV `_load_huggingface` path.
    _task = Task.from_str(task) if isinstance(task, str) else task

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

    model: CVModelAdapter | NLPModelAdapter = loader(
        model_id=model_id,
        model_path=model_path,
        task=task,
        api_url=api_url,
        device=device,
        knowledge=Knowledge.White if _task == Task.Classification else Knowledge.Black
    )
    if num_classes is not None and hasattr(model, "num_classes"):
        model.num_classes = num_classes
    model = model.to(device)
    model.eval()
    return model
