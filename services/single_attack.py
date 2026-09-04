import torch
from fastapi import APIRouter, Body, Query, HTTPException
from pydantic import ValidationError

from models import SingleAttackOutput, SingleAttackProps, ModelInfo, RegisteredObject
from nn_trust import Task, EvasionAttack, AttackFactory as AF
from services.utils.attack import single_attack_performance
from services.utils.utils import b64str_to_pil
from utils import load_model

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

    device: torch.device = torch.device(device if device in ["cpu", "cuda", "mps"] else "cpu")

    ################## MODEL ##################
    try:
        # Extracting the values from the body
        model_info: ModelInfo = body.model
        attack: RegisteredObject = body.attack
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
    model = model.to(device)
    model.eval()
    print(" Model loaded ".center(40, "#"))

    ################## ATTACK ##################
    atk: RegisteredObject = body.attack
    attack: EvasionAttack = AF.create(
        model=model.to(device),
        class_id=atk.id,
        task=task,
        **{
            param.id: param.default
            for param in atk.parameters
        }
    )
    print(" Attack Created ".center(40, "#"))
    ############################################
    out: SingleAttackOutput = single_attack_performance(
        model=model,
        attack=attack,
        task=task,
        pil_image=b64str_to_pil(body.input),
        input_dimensionality=model_info.input_dimensionality,
        transformation=model_info.transformation,
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
            description="The device to run the model on.",
            examples=["cpu"]
        )

) -> dict:
    """
    Handle the POST request for executing a jailbreak attack.
    """
    if device in ["cpu", "cuda"]:
        device = torch.device(device)
    else:
        device = torch.device("cpu")

    model_info = body.get("model")
    attack_info = body.get("attack")
    goal = body.get("input")
    attacker_info = body.get("attacker")
    judge_info = body.get("judge")

    # ── 1. Load models ──────────────────────────────────────────────────────
    def _load_nlp_model(info: dict):
        """Load an NLP model adapter from its info dict."""
        m = load_model(
            model_type=info.get("model_type", "HuggingFace"),
            model_path=info.get("repository"),
            task=Task.from_str(info.get("task", "language")),
            model_api=info.get("api"),
            model_id=info.get("id"),
        )
        if hasattr(m, "model") and hasattr(m.model, "parameters"):
            m = m.to(device)
            m.eval()
        return m

    def _load_if_provided(info: dict | None, fallback_model, fallback_info: dict):
        if info is None or info.get("id") == fallback_info.get("id"):
            return fallback_model
        return _load_nlp_model(info)

    # Target model (always uses the route model from the store)
    target_model = _load_nlp_model(model_info)

    # Attacker and judge — fall back to target when not provided or same ID
    attacker_model = _load_if_provided(attacker_info, target_model, model_info)
    judge_model = _load_if_provided(judge_info, target_model, model_info)

    # ── 2. Instantiate the attack ───────────────────────────────────────────
    kwargs = {param.get("id"): param.get("default") for param in attack_info.get("parameters", [])}
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

    # 4. Build response from ConversationState (now a dataclass, not Pydantic)
    # The ConversationState has: goal, success, best_response, best_score,
    # attempts (list[AttackAttempt]), metadata, stateful flag, etc.

    # Derive best_prompt from the highest-scored attempt
    best_prompt = ""
    scored_attempts = [a for a in state.attempts if a.score is not None]
    if scored_attempts:
        best_attempt = max(scored_attempts, key=lambda a: a.score)
        best_prompt = best_attempt.prompt

    if state.stateful:
        # Stateful attack (e.g. Red Queen): one continuous conversation.
        # The target_context holds the full dialogue (system, user, assistant).
        conversations = [[
            {
                "role": "attacker" if m.role == "user" else "target",
                "content": m.content,
                "score": None,
            }
            for m in state.target_context
            if m.role != "system"  # exclude internal system prompt from display
        ]]
        # Flat history == same as the single conversation (no separate attempts to show)
        history = conversations[0]
    else:
        # Stateless attack (PAIR, GCG, Prefill, …): each attempt is an
        # independent target query that starts a fresh conversation.
        conversations = []
        for attempt in state.attempts:
            conversations.append([
                {"role": "attacker", "content": attempt.prompt, "score": attempt.score},
                {"role": "target", "content": attempt.response, "score": attempt.score},
            ])
        # Flat history: all turns concatenated (for the "full history" view)
        history = [turn for chat in conversations for turn in chat]

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
