from transformers import AutoModelForCausalLM, AutoTokenizer

from nn_trust import NLPModelAdapter, Knowledge, Task
from nn_trust.attack.nlp.adapters import HuggingFaceNLPAdapter, OllamaNLPAdapter


def _load_ollama(
        model_id: str,
        api_url: str,
        **kwargs
) -> NLPModelAdapter:
    # Ollama models are always remote LLMs and never go through the CV path.
    if model_id is None:
        raise ValueError("model_id is required for Ollama models.")
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
