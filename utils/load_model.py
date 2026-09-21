from pathlib import Path
import os
import json
import weakref
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
    JailJudgeGuard,
    LlamaGuardJudge,
    LlamaGuardJudgeWithCategories,
    Qwen3GuardJudge,
    Qwen3GuardStreamJudge,
    WildGuardJudge,
    WildGuardLogitJudge,
    GraniteGuardianJudge,
    GraniteGuardianLogitJudge,
    RewardAnythingJudge,
)
from nn_trust.attack.nlp.judge import LLMJudge

try:
    from llama_cpp import Llama
except ImportError:  # llama_cpp optional: only needed for model_type="Llamacpp"
    Llama = None


# ---------------------------------------------------------------------
#  Shared local-LLM registry
#  --------------------------------------------------------------------
#  Ollama deduplicates the same model server-side, but Llamacpp and
#  HuggingFace adapters hold full weights in this process, so loading the
#  same underlying model twice (e.g. the same model used as attacker,
#  target and judge) would allocate two full copies in VRAM.
#
#  These weak-value registries let every  load_model()  call for the same
#  model identity reuse a single shared instance.  Because the dict only
#  holds weak references, an entry dies automatically as soon as the last
#  adapter stops referencing it -- VRAM is never pinned across requests
#  and no explicit ref-counting/eviction is needed.  Use
#  clear_model_cache() to drop entries eagerly.
_HFLLM_CACHE: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
_HFLLM_TOKENIZERS: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
_LLAMACPP_CACHE: weakref.WeakValueDictionary = weakref.WeakValueDictionary()

# Adapter-level knobs that must NOT participate in a Llamacpp instance's
# identity: they tune sampling / thinking behaviour of the adapter itself,
# while n_ctx / n_gpu_layers / logits_all / verbose / task / build kwargs
# change the underlying llama.cpp context and so ARE part of the key.
_LLAMA_ADAPTER_KWARGS = frozenset({
    "name", "threat_model", "temperature", "max_new_tokens",
    "enable_thinking", "keep_thinking", "device",
})


def _hashable(v) -> bool:
    try:
        hash(v)
        return True
    except TypeError:
        return False


def _resolve_gguf(path: str | Path) -> str:
    """Resolve a GGUF file path, globbing the first .gguf inside a directory.
    Mirrors the resolution done inside LlamacppModelAdapter so the cache key
    always points at the actual file that gets loaded.  Results are
    symlink-resolved (HF snapshot dirs symlink to blobs/, and the same file
    must produce the same identity regardless of how it is reached)."""
    p = Path(path)
    if p.is_dir():
        gguf_files = list(p.glob("*.gguf"))
        if not gguf_files:
            raise FileNotFoundError(f"No .gguf file found in directory: {path}")
        return str(gguf_files[0].resolve())
    return str(p.expanduser().resolve())


def clear_model_cache() -> None:
    """Drop all shared model entries. VRAM is freed once no adapter holds a
    reference to them anymore (callers with live adapters keep working)."""
    _HFLLM_CACHE.clear()
    _HFLLM_TOKENIZERS.clear()
    _LLAMACPP_CACHE.clear()


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
            api_key=kwargs.pop("api_key", None) or os.environ.get("GEMINI_API_KEY"),
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
        resolved_file = _resolve_gguf(path_to_load)
        _ctx = kwargs.get("n_ctx", 16384)
        _layers = kwargs.get("n_gpu_layers", -1)
        _logits_all = kwargs.get("logits_all", True)
        _verbose = kwargs.get("verbose", False)
        _build = frozenset(
            (k, v) for k, v in kwargs.items()
            if k not in _LLAMA_ADAPTER_KWARGS
            and k not in {"n_ctx", "n_gpu_layers", "logits_all", "verbose"}
            and _hashable(v)
        )

        _key = (resolved_file, _ctx, _layers, _logits_all, _verbose, _build)
        llama = _LLAMACPP_CACHE.get(_key)
        if llama is None:
            # Only the identity-defining params (the same ones captured in
            # _key) go into the build; adapter-level knobs stay on the
            # wrapper.  non-hashable extra kwargs are intentionally omitted
            # from the identity (they are exotic / rarely used).
            llama = Llama(
                model_path=resolved_file,
                n_ctx=_ctx,
                n_gpu_layers=_layers,
                logits_all=_logits_all,
                verbose=_verbose,
                **dict(_build),
            )
            _LLAMACPP_CACHE[_key] = llama
        model = LlamacppModelAdapter(
            model_path=resolved_file,
            name=model_id or str(resolved_path),
            task=task or Task.Language,
            llama_instance=llama,
            **kwargs,
        )

    _task = Task.from_str(task) if isinstance(task, str) else task
    if model_type == "HuggingFace" and _task == Task.Language:
        if model_id is None:
            raise ValueError("model_id is required for HuggingFace LLMs.")
        # Share one (model, tokenizer) pair per model_id so repeated loads
        # of the same LLM (attacker == target == judge, ...) don't allocate
        # separate fp32 copies in VRAM.  Entries are weak: they disappear
        # once the last adapter stops referencing the model.
        llm = _HFLLM_CACHE.get(model_id)
        tok = _HFLLM_TOKENIZERS.get(model_id)
        if llm is None or tok is None:
            llm = AutoModelForCausalLM.from_pretrained(model_id)
            tok = AutoTokenizer.from_pretrained(model_id)
            _HFLLM_CACHE[model_id] = llm
            _HFLLM_TOKENIZERS[model_id] = tok
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
        if judge_type == "jailjudge":
            return JailJudgeGuard(adapter=model, **judge_kwargs)
        elif judge_type == "llama_guard":
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
        elif judge_type == "granite_guardian":
            return GraniteGuardianJudge(adapter=model, **judge_kwargs)
        elif judge_type == "granite_guardian_logit":
            return GraniteGuardianLogitJudge(adapter=model, **judge_kwargs)
        elif judge_type == "reward_anything":
            return RewardAnythingJudge(adapter=model, **judge_kwargs)
        else:
            return LLMJudge(adapter=model, **judge_kwargs)

    return model
