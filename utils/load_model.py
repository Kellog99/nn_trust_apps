from pathlib import Path
from typing import Callable, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.info import MODEL_TYPES
from nn_trust import CVModelAdapter, Task, Knowledge
from nn_trust.attack.nlp.adapters import HuggingFaceNLPAdapter, OllamaNLPAdapter
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
    # ── LLM loading (NLP adapters) ───────────────────────────────────────
    # Ollama models are always remote LLMs and never go through the CV path.
    if model_type == "Ollama":
        if model_id is None:
            raise ValueError("model_id is required for Ollama models.")
        return OllamaNLPAdapter(
            model_id=model_id,
            base_url=api_url or "http://localhost:11434",
            name=model_id,
            **kwargs,
        )

    # HuggingFace can be a CV model (local checkpoint via the CV path below)
    # or an LLM (hub causal LM). Ambiguity is resolved by task: a Language
    # task selects the HuggingFaceNLPAdapter; anything else falls through to
    # the unchanged CV `_load_huggingface` path.
    _task = Task.from_str(task) if isinstance(task, str) else task
    if model_type == "HuggingFace" and _task == Task.Language:
        if model_id is None:
            raise ValueError("model_id is required for HuggingFace LLMs.")
        llm = AutoModelForCausalLM.from_pretrained(model_id)
        tok = AutoTokenizer.from_pretrained(model_id)
        return HuggingFaceNLPAdapter(
            model=llm,
            tokenizer=tok,
            name=model_id,
            threat_model=Knowledge.White,
            task=Task.Language,
            **kwargs,
        )

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
    if num_classes is not None:
        model.num_classes = num_classes
    model = model.to(device)
    model.eval()
    return model
