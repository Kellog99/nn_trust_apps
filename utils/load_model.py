from pathlib import Path
import os
import json
from typing import Callable, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.info import MODEL_TYPES
from nn_trust import CVModelAdapter, Task, Knowledge
from nn_trust.attack.nlp.adapters import (
    HuggingFaceNLPAdapter,
    OllamaNLPAdapter,
    GeminiAIStudioAdapter,
    OpenAINLPAdapter,
    LlamacppModelAdapter,
)
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
from nn_trust.attack.nlp.judges import (
    LlamaGuardJudge,
    LlamaGuardJudgeWithCategories,
    Qwen3GuardJudge,
    Qwen3GuardStreamJudge,
    WildGuardJudge,
    WildGuardLogitJudge,
)
from nn_trust.attack.nlp.judge import LLMJudge

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
    Load a model from disk or remote repository and wrap it into the shared
    `CVModelAdapter` or `NLPModelAdapter` interface. If `is_judge=True` is set
    in `info.json` or passed as a keyword argument, the model adapter is
    automatically wrapped into the appropriate judge based on `judge_type`
    (e.g., 'llama_guard', 'qwen3_guard', 'wildguard', 'llm_judge').

    Args:
        model_id
        model_type: type of model to load (e.g. "HuggingFace", "Llamacpp", "Ollama").
        model_path: Directory containing the eventual model to load.
        api_url:
        task
        num_classes:
        device: Target device. Defaults to CUDA if available, else CPU.
        **kwargs: Overrides merged into the fields loaded from
            `info.json` (e.g. `num_classes=`, `task=`, `is_judge=`, `judge_type=`).

    Returns:
        CVModelAdapter | NLPModelAdapter | BaseJudge: A unified adapter
        or judge wrapping the loaded model.
    """
    # ── Check for judge metadata in info.json if model_path is provided ──
    is_judge = kwargs.pop("is_judge", False)
    judge_type = kwargs.pop("judge_type", None)

    resolved_path = None
    if model_path is not None:
        resolved_path = Path(model_path).expanduser().resolve() if isinstance(model_path, str) else model_path
        info_file = resolved_path / "info.json"
        if info_file.exists():
            try:
                with open(info_file, "r") as f:
                    info_data = json.load(f)
                is_judge = info_data.get("is_judge", is_judge)
                judge_type = info_data.get("judge_type", judge_type)
            except Exception:
                pass

    # ── LLM loading (NLP adapters) ───────────────────────────────────────
    model = None

    if model_type == "Ollama":
        if model_id is None:
            raise ValueError("model_id is required for Ollama models.")
        model = OllamaNLPAdapter(
            model_id=model_id,
            base_url=api_url or "http://localhost:11434",
            name=model_id,
            **kwargs,
        )

    elif model_type == "Gemini":
        if model_id is None:
            raise ValueError("model_id is required for Gemini models.")
        model = GeminiAIStudioAdapter(
            model_id=model_id,
            base_url=api_url or "https://generativelanguage.googleapis.com",
            api_key=kwargs.pop("api_key", None),
            name=model_id,
            **kwargs,
        )

    elif model_type == "OpenRouter":
        if model_id is None:
            raise ValueError("model_id is required for OpenRouter models.")
        model = OpenAINLPAdapter(
            model_id=model_id,
            base_url=api_url or "https://openrouter.ai/api",
            api_key=kwargs.pop("api_key", None) or os.environ.get("OPENROUTER_API_KEY"),
            name=model_id,
            **kwargs,
        )

    elif model_type == "Llamacpp":
        path_to_load = str(resolved_path or model_id)
        if not path_to_load:
            raise ValueError("model_id or model_path is required for Llamacpp models.")
        model = LlamacppModelAdapter(
            model_path=path_to_load,
            name=model_id or str(resolved_path),
            task=task or Task.Language,
            **kwargs,
        )

    _task = Task.from_str(task) if isinstance(task, str) else task
    if model_type == "HuggingFace" and _task == Task.Language:
        if model_id is None:
            raise ValueError("model_id is required for HuggingFace LLMs.")
        llm = AutoModelForCausalLM.from_pretrained(model_id)
        tok = AutoTokenizer.from_pretrained(model_id)
        model = HuggingFaceNLPAdapter(
            model=llm,
            tokenizer=tok,
            name=model_id,
            threat_model=Knowledge.White,
            task=Task.Language,
            **kwargs,
        )

    elif model is None:
        if resolved_path is None:
            raise ValueError("model_path is required if not using remote repos.")

        try:
            loader = _LOADERS[model_type]
        except KeyError:
            raise ValueError(
                f"Unsupported model type: {model_type}. "
                f"Supported types: {sorted(_LOADERS.keys())}"
            )

        model = loader(
            model_id=model_id,
            model_path=resolved_path,
            task=task,
            api_url=api_url,
            device=device
        )
        if num_classes is not None and hasattr(model, "num_classes"):
            model.num_classes = num_classes
        if hasattr(model, "to") and hasattr(model, "parameters"):
            model = model.to(device)
            model.eval()

    # ── Wrap into Judge if is_judge is True ───────────────────────────────
    if is_judge:
        judge_kwargs = {
            k: v for k, v in kwargs.items()
            if k in {"safe_token", "unsafe_token", "name", "apply_chat_template", "temperature"}
        }
        if judge_type == "llama_guard":
            return LlamaGuardJudge(adapter=model, **judge_kwargs)
        elif judge_type == "llama_guard_categories":
            return LlamaGuardJudgeWithCategories(adapter=model, **judge_kwargs)
        elif judge_type == "qwen3_guard":
            return Qwen3GuardJudge(adapter=model, **judge_kwargs)
        elif judge_type == "qwen3_guard_stream":
            return Qwen3GuardStreamJudge(adapter=model, **judge_kwargs)
        elif judge_type == "wildguard":
            return WildGuardJudge(adapter=model, **judge_kwargs)
        elif judge_type == "wildguard_logit":
            return WildGuardLogitJudge(adapter=model, **judge_kwargs)
        else:
            return LLMJudge(adapter=model, **judge_kwargs)

    return model
