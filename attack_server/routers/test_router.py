import datetime
import time
from pathlib import Path
from pprint import pprint
import yaml

import torch
from fastapi import APIRouter, Body, Query, HTTPException
from pydantic import ValidationError
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchvision.transforms import v2 as T, InterpolationMode

from attack_server.lib.model import RegisteredObject
from attack_server.models.attack import SingleAttackOutput, SingleAttackProps
from attack_server.models.info import ModelInfo
from attack_server.utils.model import load_model
from attack_server.utils.utils import b64str_to_pil
from nn_trust import Task
from nn_trust.attack import EvasionAttack, AttackFactory as AttackFactory


# ... existing imports ...

router = APIRouter(prefix="/test", tags=["jobs management", "jobs utils"])


# --- Single attack --- #
@router.post("/single_attack")
async def single_attack(
        body: SingleAttackProps = Body(...),
        device: str = Query(
            default="cpu",
            description="The device to run the model on.",
            example="cpu"
        )
) -> SingleAttackOutput:
    """
    This function handle the POST request for executing a single image attack given:
        1. an image: str
        2. an attack: RegisterdObject
        3. a model: ModelInfo
    Args:
        body: Body of the request
        out_path: Path for saving eventually temporary files due to the logger
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
        pprint(model_info.model_dump())
    except ValidationError as e:
        print("=== VALIDATION ERROR ===")
        print(e.json())
        raise HTTPException(status_code=422, detail=e.errors())
    except Exception as e:
        print(f"=== UNEXPECTED ERROR ===")
        print(f"Error type: {type(e)}")
        print(f"Error message: {str(e)}")
        raise

    model = load_model(
        model_type=model_info.model_type,
        model_path=model_info.repository,
        task=Task.from_str(model_info.task),
        model_api=model_info.api,
        model_id=model_info.id,
    )
    model = model.to(device)
    model.eval()
    ###########################################

    ################## IMAGE ##################
    print(f"image = {body.image}")
    pil_image = b64str_to_pil(body.image)
    input_dimensionality = model_info.input_dimensionality

    if isinstance(input_dimensionality, list):
        if len(input_dimensionality) == 3:
            input_dimensionality = input_dimensionality[1:]
        elif len(input_dimensionality) == 1:
            input_dimensionality = [input_dimensionality[0], input_dimensionality[0]]
        input_dimensionality = tuple(input_dimensionality)
    elif isinstance(input_dimensionality, int):
        input_dimensionality = (input_dimensionality, input_dimensionality)

    ############ image transformation ############
    original_input: torch.Tensor = T.ToTensor()(pil_image)
    C, H, W = original_input.shape

    transformations = T.Compose([
        T.Resize(size=input_dimensionality, interpolation=InterpolationMode.NEAREST),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    x: torch.Tensor = transformations(pil_image)
    if x.dim() == 3:
        x = x.unsqueeze(0)
    x = x.to(device)
    ###############################################

    ################## ATTACK ##################
    attack: RegisteredObject = body.attack
    attack: EvasionAttack = AttackFactory.create(
        model=model,
        class_id=attack.id,
        task=Task.from_str(model_info.task),
        **{param.id: param.default for param in attack.parameters}
    )
    ############################################

    ################## Results ##################
    y = model(x)
    labels = y.argmax(-1).tolist()
    target = AvoidOnehotTarget(num_classes=y.shape[-1])(labels).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure()
    # Logger for getting additional material
    out_path = Path(f"./tmp/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    logger = PyTorchCheckpointLogger(
        states=["conf_adversarial", "conf_original"],
        path=out_path
    )

    start = time.time()
    x_adv = attack.generate(
        x=x,
        y=target,
        logger=logger
    )
    end = time.time()

    y_adv = model(x_adv).argmax(-1)
    ssim_measure = ssim_metric(x, x_adv.cpu()).item()
    conf_original, conf_adversarial = {}, {}
    if logger:
        conf_original: dict = logger.get_logging(tag="conf_original", state="generate")
        conf_adversarial: dict = logger.get_logging(tag="conf_adversarial", state="generate")
    ############################################

    ################## Invert transform ################
    inv_transform = T.Compose([
        T.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
        ),
        T.Resize(size=(H, W), interpolation=InterpolationMode.BICUBIC),
    ])
    pert = inv_transform(x_adv.cpu() - x)
    x_adv = inv_transform(x_adv.cpu())

    return SingleAttackOutput(
        x_adv=x_adv.cpu(),
        adv_perturbation=pert.cpu(),
        original_prediction=str(labels[0]),
        adversarial_prediction=str(y_adv.item()),
        advance_metrics={
            "ssim": ssim_measure,
            "distance": torch.norm(pert, p=getattr(attack.config, "p", 2)).item(),
            "execution_time": end - start,
        },
        confidence={
            "adversarial": conf_original,
            "original": conf_adversarial,
        }
    )

@router.post("/jailbreaking")
async def jailbreaking(
        body: dict = Body(...),
        device: str = Query(
            default="cuda",
            description="The device to run the model on.",
            example="cpu"
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
    judge_model    = _load_if_provided(judge_info, target_model, model_info)

    # ── 2. Instantiate the attack ───────────────────────────────────────────
    kwargs = {param.get("id"): param.get("default") for param in attack_info.get("parameters", [])}
    attack = AttackFactory.create(
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
                {"role": "target",   "content": attempt.response, "score": attempt.score},
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

