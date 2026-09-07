import os
from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer

from nn_trust import NLPModelAdapter, Knowledge, Task
from nn_trust.attack.nlp.adapters import HuggingFaceNLPAdapter, OllamaNLPAdapter, OpenAINLPAdapter


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
