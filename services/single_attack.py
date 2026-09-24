from pprint import pprint
from typing import cast

import torch
from fastapi import APIRouter, Body, HTTPException
from pydantic import ValidationError

from models import SingleAttackOutput, SingleAttackProps, ModelInfo, RegisteredObject, JailbreakAttackProps, \
    JailbreakAttackOutput
from nn_trust import Task, EvasionAttack, AttackFactory as AF, NLPModelAdapter, CVModelAdapter
from services.utils.attack import single_attack_performance
from services.utils.utils import b64str_to_pil
from utils import load_model

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
        base64_img: str | None = body.input
        if base64_img is None:
            raise ValueError("The input image in the evasion attack cannot be None.")
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
        body: JailbreakAttackProps = Body(...),
) -> JailbreakAttackOutput:
    """
    Handle the POST request for executing a jailbreak attack.
    """
    device: torch.device = body.resolve_device()
    missing: list[str] = [n for n in ("model", "attack", "input") if getattr(body, n) is None]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Missing required field(s): {', '.join(missing)}",
        )

    attack_info: RegisteredObject = body.attack
    goal: str = body.input
    max_new_tokens = body.max_new_tokens if body.max_new_tokens is not None else 4096

    def _load_model(
            info: ModelInfo,
            max_tokens: int = 256
    ) -> NLPModelAdapter:

        m = load_model(
            model_id=info.id,
            model_type=info.model_type,
            model_path=info.repository,
            task=info.task if isinstance(info.task, Task) else Task.from_str(info.task),
            model_api=info.api,
            api_key=info.api_key,
            max_new_tokens=max_tokens,
        )
        if hasattr(m, "model") and hasattr(m.model, "parameters"):
            m = m.to(device)
            m.eval()
        return m

    # Target model (always uses the route model from the store)
    target_model = _load_model(
        info=body.model,
        max_tokens=max_new_tokens
    )

    # Attacker and judge — fall back to target when not provided or same ID
    attacker_model = _load_model(
        info=cast(ModelInfo, body.attacker),
        max_tokens=max_new_tokens
    )

    judge_model = _load_model(
        info=cast(ModelInfo, body.judge),
        max_tokens=16
    )

    # ── 2. Instantiate the attack ───────────────────────────────────────────
    kwargs = _normalize_attack_parameters(
        attack_info.id,
        {
            param.id: param.default
            for param in attack_info.parameters
        },
    )
    kwargs["verbose"] = True

    pprint(kwargs)
    attack = AF.create(
        class_id=attack_info.id,
        model=target_model,
        attacker=attacker_model,
        judge=judge_model,
        device=device,
        **kwargs
    )
    print(" Attack Created ".center(40, "#"))
    # 3. Execution
    print(" Generating the prompt ".center(40, "#"))
    state = attack.generate(goal=goal)
    print(" Prompt generated ".center(40, "#"))

    # 4. Build response from ConversationState (now a dataclass, not Pydantic)
    # The ConversationState has: goal, success, best_response, best_score,
    # attempts (list[AttackAttempt]), metadata, stateful flag, etc.

    # Derive the best prompt from the highest-scored recorded attempt.
    best_prompt = ""
    scored_attempts = [attempt for attempt in state.attempts if attempt.score is not None]
    if scored_attempts:
        best_prompt = max(scored_attempts, key=lambda attempt: attempt.score).prompt

    if state.stateful:
        # Stateful attacks keep one continuous conversation in target_context.
        conversations = [[
            {
                "role": "attacker" if message.role == "user" else "target",
                "content": message.content,
                "score": None,
            }
            for message in state.target_context
            if message.role != "system"
        ]]
        history = conversations[0]
    else:
        # Stateless attacks record independent prompt/response attempts.
        conversations = [
            [
                {"role": "attacker", "content": attempt.prompt, "score": attempt.score},
                {"role": "target", "content": attempt.response, "score": attempt.score},
            ]
            for attempt in state.attempts
        ]
        history = [turn for conversation in conversations for turn in conversation]

    return JailbreakAttackOutput(
        goal=state.goal,
        success=state.success,
        best_prompt=best_prompt,
        best_response=state.best_response or "",
        best_score=state.best_score if state.best_score != float("-inf") else 0.0,
        history=history,
        conversations=conversations,
        metadata=state.metadata,
    )
