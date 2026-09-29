import os
import weakref
from pathlib import Path
from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer
from nn_trust.attack.nlp.adapters import (
    LlamacppModelAdapter,
)

from nn_trust import NLPModelAdapter, Knowledge, Task
from nn_trust.attack.nlp.adapters import (
    HuggingFaceNLPAdapter,
    OllamaNLPAdapter,
    OpenAINLPAdapter,
    LlamacppModelAdapter
)
from llama_cpp import Llama

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


def clear_model_cache() -> None:
    """Drop all shared model entries. VRAM is freed once no adapter holds a
    reference to them anymore (callers with live adapters keep working)."""
    _HFLLM_CACHE.clear()
    _HFLLM_TOKENIZERS.clear()
    _LLAMACPP_CACHE.clear()


def _load_llamacpp(
        model_path: Path | str | None = None,
        model_id: str | None = None,
        task: Task | str | None = None,
        name: str | None = None,
        threat_model: Knowledge | str = Knowledge.Black,
        temperature: float = 0.7,
        max_new_tokens: int = 256,
        enable_thinking: bool = True,
        keep_thinking: bool = False,
        n_ctx: int = 16384,
        n_gpu_layers: int = -1,
        logits_all: bool = True,
        verbose: bool = False,
        **kwargs
) -> NLPModelAdapter:
    """Load a local GGUF model, reusing a compatible llama.cpp instance.

    Args:
        model_path: GGUF file or directory containing one. Takes precedence
            over ``model_id`` when both are provided.
        model_id: Model identifier used as the display name and, if
            ``model_path`` is absent, as the local GGUF path.
        task: Task assigned to the adapter; defaults to ``Task.Language``.
        name: Adapter display name; defaults to ``model_id`` or the path.
        threat_model: Knowledge level assigned to the adapter.
        temperature: Sampling temperature used by the adapter.
        max_new_tokens: Default maximum number of generated tokens.
        enable_thinking: Whether to enable thinking for compatible models.
        keep_thinking: Whether to retain thinking text in generated output.
        n_ctx: Context window size for the llama.cpp instance.
        n_gpu_layers: Number of layers to offload to the GPU; ``-1`` means
            all available layers.
        logits_all: Whether llama.cpp computes logits for every token.
        verbose: Whether llama.cpp prints diagnostic output.

    Returns:
        An NLP adapter backed by the loaded GGUF model. Instances with the
        same path and llama.cpp settings share the underlying model.
    """

    def _resolve_gguf(path: str | Path) -> str:
        """Resolve a GGUF file or recursively find a unique GGUF in a directory.
        The cache key always points at the actual file that gets loaded. Results are
        symlink-resolved (HF snapshot dirs symlink to blobs/, and the same file
        must produce the same identity regardless of how it is reached)."""
        p = Path(path).expanduser()
        if p.is_dir():
            gguf_files = sorted({f.resolve() for f in p.rglob("*.gguf") if f.is_file()})
            if not gguf_files:
                raise FileNotFoundError(f"No .gguf file found in directory: {path}")
            if len(gguf_files) > 1:
                raise ValueError(f"Multiple GGUF files found under {path}; provide the exact model_path.")
            return str(gguf_files[0])
        return str(p.resolve())

    path_to_load = model_path if model_path is not None else model_id
    if path_to_load is None or not str(path_to_load):
        raise ValueError("model_id or model_path is required for Llamacpp models.")
    resolved_file = _resolve_gguf(path_to_load)
    if not Path(resolved_file).is_file():
        raise FileNotFoundError(f"GGUF model file not found: {path_to_load}")
    if Llama is None:
        raise ImportError("llama-cpp-python is required for Llamacpp models.")

    key = (resolved_file, n_ctx, n_gpu_layers, logits_all, verbose)
    llama = _LLAMACPP_CACHE.get(key)
    if llama is None:
        llama = Llama(
            model_path=resolved_file,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            logits_all=logits_all,
            verbose=verbose,
        )
        _LLAMACPP_CACHE[key] = llama
    return LlamacppModelAdapter(
        model_path=resolved_file,
        name=name or model_id or str(path_to_load),
        threat_model=threat_model,
        task=task or Task.Language,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        enable_thinking=enable_thinking,
        keep_thinking=keep_thinking,
        llama_instance=llama,
    )


def _load_ollama(
        model_id: str,
        api_url: str,
        **kwargs
) -> NLPModelAdapter:
    # Ollama model are always remote LLMs and never go through the CV path.
    if model_id is None:
        raise ValueError("model_id is required for Ollama model.")
    return OllamaNLPAdapter(
        model_id=model_id,
        base_url=api_url or "http://localhost:11434",
        name=model_id,
        **kwargs,
    )


def _load_huggingface_nlp(
        model_id: str,
        knowledge: Knowledge,
        task: Task,
        **kwargs
) -> NLPModelAdapter:
    if model_id is None:
        raise ValueError("model_id is required for HuggingFace LLMs.")
    llm = AutoModelForCausalLM.from_pretrained(model_id)
    tok = AutoTokenizer.from_pretrained(model_id)
    return HuggingFaceNLPAdapter(
        model=llm,
        tokenizer=tok,
        name=model_id,
        threat_model=knowledge,
        task=task,
        **kwargs,
    )


def _load_gemini(
        model_id: str,
        model_name: Optional[str] = None,
        model_api: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs
) -> NLPModelAdapter:
    """
    Load the Gemini model using an API
    """
    if model_id is None:
        raise ValueError("model_id is required for Gemini model.")
    api_key: str | None = api_key or os.environ.get("GEMINI_API_KEY")
    if api_key is None:
        raise ValueError("The API key is required for using the Gemini model.")

    return OpenAINLPAdapter(
        model_id=model_id,
        base_url=model_api or "https://generativelanguage.googleapis.com",
        api_key=api_key,
        name=model_name or model_id,
        **kwargs,
    )


def _load_openrouter(
        model_id: str,
        model_name: Optional[str] = None,
        model_api: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs
) -> NLPModelAdapter:
    """
    Load the open router model using an API
    """
    if model_id is None:
        raise ValueError("model_id is required for the OpenRouter models.")
    api_key: str | None = api_key or os.environ.get("OPENROUTER_API_KEY")
    if api_key is None:
        raise ValueError("The API key is required for using the OpenRouter model.")

    return OpenAINLPAdapter(
        model_id=model_id,
        base_url=model_api or "https://openrouter.ai/api",
        api_key=api_key,
        name=model_name or model_id,
        **kwargs,
    )
