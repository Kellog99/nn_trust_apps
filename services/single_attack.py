import logging

import torch
from fastapi import APIRouter, Body, Query, HTTPException
from pydantic import ValidationError

from models import SingleAttackOutput, SingleAttackProps, ModelInfo, RegisteredObject, JailbreakHistoryEntry
from nn_trust import Task, EvasionAttack, AttackFactory as AF, NLPModelAdapter, CVModelAdapter
from nn_trust.attack import (
    save_conversation_state,
    load_conversation_state,
    list_conversation_states,
    delete_conversation_state,
)
from nn_trust.attack.nlp import ConversationState, NLPAttack
from services.utils.attack import single_attack_performance
from services.utils.utils import b64str_to_pil
from utils import load_model

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/test", tags=["jobs management", "jobs utils"])


def _normalize_attack_parameters(attack_id: str | None, parameters: dict) -> dict:
    """Make attack parameters compatible with the optimizer constraints.

    Older clients can submit FOM's ``nesterov`` flag without updating the
    momentum field from its default value of zero.  PyTorch's SGD requires
    positive momentum and zero dampening when Nesterov acceleration is used.
    Preserve the requested Nesterov mode while filling in those constraints.
    """
    if (attack_id or "").lower() == "fom" and parameters.get("nesterov") is True:
        if float(parameters.get("momentum", 0)) <= 0:
            parameters["momentum"] = 0.9
        parameters["dampening"] = 0
    return parameters


# --- Single attack --- #
@router.post("/single_attack")
async def single_attack(
        body: SingleAttackProps = Body(...),
) -> SingleAttackOutput:
    """
    This function handle the POST request for executing a single image attack given:
        1. an image: str
        2. an attack: RegisteredObject
        3. a model: ModelInfo
    Args:
        body: Body of the request

    Returns:
        SingleAttackOutput: a collection of all the results concerning a single attack.
    """
    device = body.resolve_device()
    print(f"device = {device}")
    ################## MODEL ##################
    try:
        # Extracting the values from the body
        model_info: ModelInfo = body.model
        atk: RegisteredObject = body.attack
    except ValidationError as e:
        print("=== VALIDATION ERROR ===")
        print(e.json())
        raise HTTPException(status_code=422, detail=e.errors())
    except Exception as e:
        print(f"=== UNEXPECTED ERROR ===")
        print(f"Error type: {type(e)}")
        print(f"Error message: {str(e)}")
        raise

    task = model_info.task
    if task is None:
        raise ValueError("The task cannot be None.")
    elif isinstance(task, str):
        task: Task = Task.from_str(task)

    model = load_model(
        model_type=model_info.model_type,
        model_path=model_info.repository,
        task=task,
        model_api=model_info.api,
        model_id=model_info.id,
    )
    if isinstance(model, NLPModelAdapter):
        raise ValidationError("The model has been validated as an NLP model while it should be a CV model.")
    model: CVModelAdapter = model.to(device)
    model.eval()
    print(" Model loaded ".center(40, "#"))

    ################## ATTACK ##################
    attack_parameters = _normalize_attack_parameters(
        atk.id,
        {param.id: param.default for param in atk.parameters if param.id != "device"},
    )
    attack: EvasionAttack = AF.create(
        model=model.to(device),
        class_id=atk.id,
        task=task,
        device=device,
        **attack_parameters
    )
    print(" Attack Created ".center(40, "#"))
    ############################################
    out: SingleAttackOutput = single_attack_performance(
        model=model,
        attack=attack,
        task=task,
        transformation=model_info.transformation,
        pil_image=b64str_to_pil(body.input),
        input_dimensionality=model_info.input_dimensionality,
        device=device
    )
    print(out.confidence, out.advance_metrics)
    return out

    # # --- Single attack --- #


@router.post("/jailbreaking")
async def jailbreaking(
        body: dict = Body(...),
        device: str = Query(
            default="cuda",
            description="The device to run the model on."
        )

) -> dict:
    """
    Handle the POST request for executing a jailbreak attack.
    """
    device = torch.device(device if device in ["cpu", "cuda", "mps"] else "cpu")

    model_info = body.get("model")
    attack_info = body.get("attack")
    goal = body.get("input")
    attacker_info = body.get("attacker")
    judge_info = body.get("judge")
    max_new_tokens = body.get("max_new_tokens", 2048)

    missing = [
        name for name, value in (
            ("model", model_info),
            ("attack", attack_info),
            ("input", goal),
        )
        if value is None
    ]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Missing required field(s): {', '.join(missing)}",
        )

    # ── 1. Load model ──────────────────────────────────────────────────────
    def _load_nlp_model(
            info: dict,
            max_tokens: int = 256
    ) -> NLPModelAdapter:
        """Load an NLP model adapter from its info dict."""
        m = load_model(
            model_type=info.get("model_type", "HuggingFace"),
            model_path=info.get("repository"),
            task=Task.from_str(info.get("task", "language")),
            model_api=info.get("api"),
            model_id=info.get("id"),
            api_key=info.get("api_key") or info.get("key"),
            max_new_tokens=max_tokens,
        )
        if hasattr(m, "model") and hasattr(m.model, "parameters"):
            m = m.to(device)
            m.eval()
        return m

    def _load_if_provided(
            info: dict | None,
            fallback_model,
            fallback_info: dict,
            max_tokens: int = 256
    ) -> NLPModelAdapter:
        if info is None or info.get("id") == fallback_info.get("id"):
            return fallback_model
        return _load_nlp_model(info, max_tokens=max_tokens)

    # Target model (always uses the route model from the store)
    target_model = _load_nlp_model(model_info, max_tokens=max_new_tokens)

    # Attacker and judge — fall back to target when not provided or same ID
    attacker_model = _load_if_provided(attacker_info, target_model, model_info, max_tokens=max_new_tokens)
    judge_model = _load_if_provided(judge_info, target_model, model_info, max_tokens=16)

    # ── 2. Instantiate the attack ───────────────────────────────────────────
    kwargs = _normalize_attack_parameters(
        attack_info.get("id"),
        {param.get("id"): param.get("default") for param in attack_info.get("parameters", [])},
    )
    attack = AF.create(
        class_id=attack_info.get("id"),
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
        save_conversation_state(state, attack_id=attack_info.get("id"))
    except Exception:
        logger.exception("Failed to save conversation state for attack '%s'", attack_info.get("id"))

    # The attack knows how its own runs are shaped (e.g. TreeCrescendo returns
    # one root-to-leaf path per leaf), so let it extract the conversations.
    return _conversation_state_to_output(attack.extract_conversations(state), state)


def _conversation_state_to_output(
        conversations: list[list[dict]],
        state: ConversationState,
) -> dict:
    """
    Turn a (conversations, state) pair -- coming either from a freshly executed
    attack or from a replayed saved state -- into the payload the frontend expects.
    """
    # Build response from ConversationState (now a dataclass, not Pydantic)
    # The ConversationState has: goal, success, best_response, best_score,
    # attempts (list[AttackAttempt]), metadata, stateful flag, etc.

    # Derive the best prompt from the highest-scored recorded attempt.
    best_prompt = ""
    scored_attempts = [attempt for attempt in state.attempts if attempt.score is not None]
    if scored_attempts:
        best_prompt = max(scored_attempts, key=lambda attempt: attempt.score).prompt

    history = [turn for conversation in conversations for turn in conversation]

    ret: dict = {
        "goal": state.goal,
        "success": state.success,
        "best_prompt": best_prompt,
        "best_response": state.best_response or "",
        "best_score": state.best_score if state.best_score != float("-inf") else 0.0,
        "history": history,
        "conversations": conversations,
        "metadata": state.metadata,
    }

    return ret


# --- Jailbreak attack history (saved states board) --- #
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


@router.get("/jailbreaking/history/{attack_id}/{save_id}")
async def jailbreaking_history_replay(
        attack_id: str,
        save_id: str,
) -> dict:
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
