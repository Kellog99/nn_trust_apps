import logging
import time
import torch
from fastapi import APIRouter, Body, Query, HTTPException
from pydantic import ValidationError, BaseModel

from typing import Optional

from models import SingleAttackOutput, SingleAttackProps, JailbreakAttackProps, JailbreakAttackOutput, JailbreakHistoryEntry, Bubble, ModelInfo, RegisteredObject
from nn_trust import Task
from nn_trust.attack import (
    EvasionAttack,
    AttackFactory as AF,
    save_conversation_state,
    load_conversation_state,
    list_conversation_states,
    delete_conversation_state,
)
from nn_trust.attack.nlp import ConversationState, NLPAttack
from services.utils.attack import single_attack_performance
from services.utils.utils import b64str_to_pil
from utils import load_model

from pprint import pprint

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/test", tags=["jobs management", "jobs utils"])


# --- Single attack --- #
@router.post("/single_attack")
async def single_attack(
        body: SingleAttackProps = Body(...),
        device: str = Query(
            default="cpu",
            description="The device to run the model on."
        )
) -> SingleAttackOutput:
    """
    This function handle the POST request for executing a single image attack given:
        1. an image: str
        2. an attack: RegisteredObject
        3. a model: ModelInfo
    Args:
        body: Body of the request
        device: device where the computations are done.

    Returns:
        SingleAttackOutput: a collection of all the results concerning a single attack.
    """

    if device in ["cpu", "cuda"]:
        device = torch.device(device)
    else:
        device = torch.device("cpu")

    ################## MODEL ##################
    try:
        # Your existing code...
        model_info: ModelInfo = body.model
    except ValidationError as e:
        print("=== VALIDATION ERROR ===")
        print(e.json())
        raise HTTPException(status_code=422, detail=e.errors())
    except Exception as e:
        print(f"=== UNEXPECTED ERROR ===")
        print(f"Error type: {type(e)}")
        print(f"Error message: {str(e)}")
        raise

    task = Task.from_str(model_info.task),

    model = load_model(
        model_type=model_info.type,
        model_path=model_info.repository,
        task=task,
        model_api=model_info.api,
        model_id=model_info.id,
    )
    model = model.to(device)
    model.eval()
    print(" Model loaded ".center(40, "#"))

    ################## ATTACK ##################
    atk: RegisteredObject = body.attack
    attack: EvasionAttack = AF.create(
        model=model.to(device),
        class_id=atk.id,
        task=task,
        **{param.id: param.default for param in attack.parameters}
    )
    print(" Attack Created ".center(40, "#"))
    ############################################

    return single_attack_performance(
        model=model,
        attack=attack,
        pil_image=b64str_to_pil(body.input),
        input_dimensionality=model_info.input_dimensionality,
        device=device
    )


@router.post("/jailbreaking")
async def jailbreaking(
        body: JailbreakAttackProps = Body(...),
        device: str = Query(
            default="cuda",
            description="The device to run the model on.",
            example="cpu"
        )

) -> JailbreakAttackOutput:
    """
    Handle the POST request for executing a jailbreak attack.
    """
    if device in ["cpu", "cuda"]:
        device = torch.device(device)
    else:
        device = torch.device("cpu")

    model_info = body.model
    attack_info = body.attack
    goal = body.input
    attacker_info = body.attacker
    judge_info = body.judge
    max_new_tokens = body.max_new_tokens
    n_ctx = body.n_ctx

    # ── 1. Load models ──────────────────────────────────────────────────────
    def _load_nlp_model(info: ModelInfo | dict, max_tokens: int = 256):
        """Load an NLP model adapter from its info object or dict."""
        if info is None:
            return None
        if isinstance(info, BaseModel):
            model_type = getattr(info, "model_type", "HuggingFace")
            repository = info.repository
            task_val = info.task
            api = info.api
            model_id = info.id
            api_key = getattr(info, "api_key", None) or getattr(info, "key", None)
            device_override = getattr(info, "device", None)
        else:
            model_type = info.get("model_type", "HuggingFace")
            repository = info.get("repository")
            task_val = info.get("task", "language")
            api = info.get("api")
            model_id = info.get("id")
            api_key = info.get("api_key") or info.get("key")
            device_override = info.get("device")

        task = Task.from_str(task_val) if isinstance(task_val, str) else task_val
        # A model can override the request-level device (e.g. force the judge
        # onto CPU while attacker/target stay on GPU when VRAM is tight).
        model_device = torch.device(device_override) if device_override else device

        load_kwargs = dict(
            model_type=model_type,
            model_path=repository,
            task=task,
            model_api=api,
            model_id=model_id,
            api_key=api_key,
            max_new_tokens=max_tokens,
            # ModelInfo carries its own judge metadata; forward it so API / Ollama
            # judges (which have no info.json on disk to read) still get wrapped
            # into the right judge instead of silently degrading to a raw adapter.
            is_judge=(
                bool(getattr(info, "is_judge", False))
                if isinstance(info, BaseModel)
                else bool(info.get("is_judge", False))
            ),
            judge_type=(
                getattr(info, "judge_type", None)
                if isinstance(info, BaseModel)
                else info.get("judge_type")
            ),
        )
        # Llamacpp (GGUF) adapters use n_ctx for the context window; only
        # pass it when supplied so HuggingFace adapters don't choke on it.
        if model_type == "Llamacpp":
            if n_ctx:
                load_kwargs["n_ctx"] = n_ctx
            # llama.cpp offloads every layer to GPU by default regardless of
            # the request device; honour a CPU override explicitly.
            if model_device.type == "cpu":
                load_kwargs["n_gpu_layers"] = 0

        m = load_model(**load_kwargs)
        if hasattr(m, "model") and hasattr(m.model, "parameters"):
            m = m.to(model_device)
            m.eval()
        return m

    def _load_if_provided(info: Optional[ModelInfo | dict], fallback_model, fallback_info: ModelInfo | dict, max_tokens: int = 256):
        fallback_id = fallback_info.id if isinstance(fallback_info, BaseModel) else fallback_info.get("id")
        info_id = info.id if isinstance(info, BaseModel) else info.get("id") if info else None
        if info is None or info_id == fallback_id:
            return fallback_model
        return _load_nlp_model(info, max_tokens=max_tokens)

    # Target model (always uses the route model from the store)
    target_model = _load_nlp_model(model_info, max_tokens=max_new_tokens)

    # Attacker and judge — fall back to target when not provided or same ID
    attacker_model = _load_if_provided(attacker_info, target_model, model_info, max_tokens=max_new_tokens)
    judge_model    = _load_if_provided(judge_info, target_model, model_info, max_tokens=16)

    # ── 2. Instantiate the attack ───────────────────────────────────────────
    params = attack_info.parameters if hasattr(attack_info, "parameters") else attack_info.get("parameters", [])
    kwargs = {}
    for param in params:
        if isinstance(param, BaseModel):
            kwargs[param.id] = param.default
        else:
            kwargs[param.get("id")] = param.get("default")

    attack_id = attack_info.id if hasattr(attack_info, "id") else attack_info.get("id")

    attack = AF.create(
        class_id=attack_id,
        model=target_model,
        attacker=attacker_model,
        judge=judge_model,
        verbose=True,
        device=device,
        **kwargs
    )

    # 3. Execution
    state = attack.generate(goal=goal)

    # 4. Persist the run so it can be replayed later from the "past attacks" board.
    # Saving must never break a successful attack response.
    try:
        save_conversation_state(state, attack_id=attack_id)
    except Exception:
        logger.exception("Failed to save conversation state for attack '%s'", attack_id)

    return _conversation_state_to_output(attack.extract_conversations(state), state)


def _conversation_state_to_output(
        conversations: list[list[dict]],
        state: ConversationState,
) -> JailbreakAttackOutput:
    """
    Turn a (conversations, state) pair -- coming either from a freshly executed
    attack or from a replayed saved state -- into the payload the frontend expects.
    """
    # Derive best_prompt, best_response, and best_score from valid conversation paths
    best_prompt = ""
    best_response = state.best_response or ""
    best_score = state.best_score if state.best_score != float("-inf") else 0.0

    max_score = float("-inf")
    for chat in conversations:
        for idx, turn in enumerate(chat):
            score = turn.get("score")
            if score is not None and score > max_score:
                max_score = score
                best_score = max_score
                if turn["role"] == "target":
                    best_response = turn["content"]
                    if idx > 0 and chat[idx - 1]["role"] == "attacker":
                        best_prompt = chat[idx - 1]["content"]
                elif turn["role"] == "attacker":
                    best_prompt = turn["content"]
                    if idx + 1 < len(chat) and chat[idx + 1]["role"] == "target":
                        best_response = chat[idx + 1]["content"]

    history = [turn for chat in conversations for turn in chat]

    return JailbreakAttackOutput(
        goal=state.goal,
        success=state.success,
        best_prompt=best_prompt,
        best_response=state.best_response or "",
        best_score=state.best_score if state.best_score != float("-inf") else 0.0,
        history=history,
        conversations=conversations,
        metadata=state.metadata,
        adversarial_prompt=best_prompt,
        model_response=state.best_response or "",
    )


# --- Jailbreak attack history (saved states board) --- #
@router.get("/jailbreaking/history")
async def jailbreaking_history(
        attack_id: str = Query(..., description="Registered attack id whose saved runs to list."),
) -> list[JailbreakHistoryEntry]:
    """
    List the saved runs available for `attack_id`, most recent first, so the
    frontend can offer them as a "past attacks" board.
    """
    try:
        entries = list_conversation_states(attack_id)
    except Exception:
        logger.exception("Failed to list saved states for attack '%s'", attack_id)
        raise HTTPException(status_code=500, detail="Failed to list saved attack states.")

    return [JailbreakHistoryEntry(**entry) for entry in entries]


class _StatelessNLPAttack(NLPAttack):
    """
    Minimal concrete `NLPAttack` used only to call the (state-only)
    `extract_conversations` method on a saved state without a live model.
    """

    def step(self, i, state, **kwargs):
        raise NotImplementedError


def _attack_class_for_replay(attack_id: str) -> type:
    """Resolve the registered attack class for `attack_id`, falling back to
    the base extraction logic if it isn't a known NLP attack."""
    try:
        cls = AF.get_info(attack_id).class_type
        if issubclass(cls, NLPAttack):
            return cls
    except Exception:
        pass
    return _StatelessNLPAttack


@router.get("/jailbreaking/history/{attack_id}/{save_id}")
async def jailbreaking_history_replay(
        attack_id: str,
        save_id: str,
) -> JailbreakAttackOutput:
    """
    Load a previously saved run for `attack_id` and return it in the same
    shape as `/jailbreaking`, so the frontend can display it as if the attack
    had just been executed.
    """
    try:
        state = load_conversation_state(attack_id=attack_id, save_id=save_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # `extract_conversations` only relies on `state`, never on the attack's own
    # configuration (model/attacker/judge) -- so we can call it on an
    # uninitialized instance of the registered attack class instead of
    # reloading live models just to replay a saved conversation.
    attack_cls = _attack_class_for_replay(attack_id)
    conversations = attack_cls.extract_conversations(object.__new__(attack_cls), state)

    return _conversation_state_to_output(conversations, state)


@router.delete("/jailbreaking/history/{attack_id}/{save_id}")
async def jailbreaking_history_delete(
        attack_id: str,
        save_id: str,
) -> dict:
    """
    Delete one saved run of `attack_id` from the "past attacks" board.
    """
    try:
        delete_conversation_state(attack_id=attack_id, save_id=save_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception:
        logger.exception("Failed to delete saved state '%s/%s'", attack_id, save_id)
        raise HTTPException(status_code=500, detail="Failed to delete the saved attack state.")

    return {"deleted": save_id}
