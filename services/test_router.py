import time
import torch
from fastapi import APIRouter, Body, Query, HTTPException
from pydantic import ValidationError

from models import SingleAttackOutput, SingleAttackProps, JailbreakAttackOutput, Bubble, ModelInfo, RegisteredObject
from nn_trust import Task
from nn_trust.attack import EvasionAttack, AttackFactory as AF
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


# # --- Single attack --- #
# @router.post("/jailbreaking")
# async def jailbreaking(
#         body: SingleAttackProps = Body(...),
#         device: str = Query(
#             default="cpu",
#             description="The device to run the model on.",
#         )
# ) -> JailbreakAttackOutput:
#     time.sleep(3)

#     return JailbreakAttackOutput(
#         adversarial_prompt=
#         """
#             Lorem ipsum dolor sit amet, consectetur adipiscing elit. Aliquam dignissim ligula eu elit vestibulum efficitur. Ut felis mauris, ullamcorper nec rutrum congue, tincidunt mollis elit. Fusce sagittis lacus in vulputate posuere. Pellentesque at enim consectetur, consectetur felis in, lacinia neque. Nulla facilisi. Maecenas eu vulputate odio. Vestibulum ut erat molestie eros ultricies sodales at a felis. In tempus quam accumsan sem gravida, non consequat nisi finibus. Nam hendrerit vulputate sem, vitae ornare libero accumsan ut. Interdum et malesuada fames ac ante ipsum primis in faucibus. Quisque suscipit libero id erat placerat, nec euismod massa finibus. Etiam sit amet vestibulum elit, et viverra elit.
#             Donec sed mi tellus. Sed non maximus nisi, eu fringilla lectus. Proin facilisis sapien vel magna consequat bibendum. Proin tempor mauris id diam sodales, vitae pharetra ipsum pretium. Etiam at dapibus urna, quis congue mauris. In in tortor enim. In sed varius eros. Proin pharetra, lorem eu lacinia faucibus, tellus neque volutpat ipsum, eget finibus nisl ex sit amet enim. Curabitur et sem sed neque congue faucibus et quis lectus. Praesent interdum varius nulla, vitae dapibus dui consectetur bibendum. In vulputate vel nisi non feugiat.
#             Integer varius laoreet leo sed molestie. Cras eu tristique dolor. Aliquam interdum purus non nunc hendrerit suscipit. Integer eget convallis lacus. Maecenas nec nulla augue. Etiam pharetra a turpis a egestas. Quisque auctor, risus id porta feugiat, neque erat porta mi, in blandit sapien augue vel turpis. Pellentesque sit amet elementum odio. Donec ligula lorem, viverra nec urna nec, sagittis vulputate elit. Cras sodales quam ex, finibus vehicula felis suscipit nec.
#             In volutpat euismod metus, at sagittis sapien rhoncus vel. Donec porta non dui at egestas. Aenean eleifend bibendum mollis. Nullam varius elementum condimentum. Morbi dictum ex non ullamcorper condimentum. Curabitur porttitor ex ut felis tincidunt viverra. In accumsan vel velit quis faucibus. Duis ultricies blandit odio a malesuada. Ut pellentesque elementum nisl a accumsan. Duis mollis erat a ex condimentum ornare. Nam sed nisi sed augue imperdiet sagittis. Sed ligula dui, tincidunt a nisi vel, gravida lobortis lorem. Aenean tincidunt magna sed mi venenatis congue. Phasellus nisi enim, fermentum et vehicula nec, lacinia porta turpis. Aenean odio metus, pulvinar eu interdum sit amet, cursus sit amet velit. Sed at justo quis orci interdum posuere nec a ligula.
#             Nunc feugiat urna vel faucibus placerat. Curabitur sem est, pulvinar vitae sollicitudin vel, vulputate eu urna. Vivamus varius sit amet odio sit amet finibus. Nunc volutpat sem purus, at ultrices diam luctus eu. Pellentesque ac consequat massa, nec faucibus leo. Mauris et nibh ut lorem dapibus faucibus eu non ex. Proin pretium rhoncus lorem, in faucibus libero porta sit amet. Nam sit amet nisi ante. Sed in arcu sit amet tellus mattis fermentum eget at ante. Nullam varius quam nisl. Donec ultricies aliquam dolor, a porta enim commodo sed. Maecenas viverra aliquam suscipit.
#             Aliquam velit mauris, commodo et turpis in, mattis luctus elit. Nunc eget consectetur purus. Nam feugiat ligula vitae rhoncus ullamcorper. Morbi aliquet neque ac pellentesque maximus. Donec nisl nulla, condimentum non suscipit vitae, mollis a nisi. Nam vitae eleifend augue. Fusce finibus lobortis dui, at tincidunt ante placerat vehicula. Mauris malesuada dui quam, non ullamcorper nisl malesuada id. Etiam condimentum arcu augue. Praesent id magna facilisis elit congue pharetra. Praesent iaculis tortor erat, sit amet ullamcorper nisl viverra eget. Nunc aliquet id odio a aliquam. Aenean metus ipsum, ullamcorper eget lorem at, condimentum faucibus dolor.
#             Integer auctor odio at nibh ultricies malesuada pellentesque quis ipsum. Aliquam pretium luctus leo sed pharetra. Nulla tristique dolor at ante convallis, in pellentesque lacus efficitur. Integer vehicula ex massa, nec pellentesque massa vehicula sodales. Fusce ultrices quam mi, blandit vehicula ligula mollis in. Sed neque est, egestas ut blandit non, pulvinar in metus. Nulla luctus fermentum vehicula. Donec ornare, lorem at finibus consequat, purus arcu fringilla ex, in posuere nibh lorem pharetra dui.
#             Phasellus eu vestibulum arcu. Fusce ac risus et ante semper dignissim. Donec ac erat sit amet odio sodales luctus. Cras ac arcu rutrum, aliquet eros eu, scelerisque nisi. Suspendisse sit amet venenatis risus. Nam non magna nulla. Phasellus tincidunt ex id dolor feugiat congue. Suspendisse convallis urna non ante eleifend, eu vulputate nisl hendrerit. Integer massa libero, cursus sit amet fermentum in, auctor id ante. Nam sit amet feugiat odio.
#             Vivamus rhoncus ligula vel enim cursus auctor. Nullam vel tempor justo. Aliquam ut sapien bibendum, vehicula odio vel, rutrum nisl. Nunc a est a arcu placerat malesuada vel non eros. Phasellus scelerisque urna ut dolor eleifend, vel facilisis purus laoreet. Donec facilisis pretium ultrices. Phasellus urna metus, congue non metus ut, pellentesque molestie urna. In blandit tellus non erat tincidunt tincidunt. Mauris sit amet fermentum velit. Ut iaculis vel mi sit amet dapibus. Curabitur nec ex dignissim, volutpat diam quis, hendrerit orci. In condimentum porta felis ut mattis. Vivamus euismod nulla eros, vel ultricies purus tempor sit amet. In hac habitasse platea dictumst. Suspendisse potenti. Nunc ultricies sapien mauris, id vulputate nisi mattis a.
#             Donec viverra mauris nisl, sed porttitor orci tincidunt ut. Quisque purus massa, hendrerit vitae quam at, pharetra semper metus. Etiam et malesuada eros. Aenean nec lacus nisl. In faucibus imperdiet sapien ac facilisis. Nunc sem purus, congue quis pharetra at, congue eget mauris. Sed accumsan sapien in nulla semper, vel laoreet nibh pulvinar. Cras finibus suscipit interdum. Nunc quis sollicitudin lacus. Sed odio libero, vulputate vulputate justo ac, ultricies maximus nisi. Pellentesque rhoncus, mi quis tristique pharetra, nibh nibh maximus ipsum, sit amet pharetra odio metus et tellus. Integer fermentum justo at vestibulum rhoncus. Suspendisse sagittis massa vitae risus ultrices, sit amet tincidunt justo pharetra. Vestibulum ultrices interdum felis, non hendrerit odio ullamcorper at. Sed aliquet lorem sed quam viverra porta nec in leo.
#         """,
#         conversations=[
#             [
#                 Bubble(
#                     sender="user" if i % 2 == 0 else "model",
#                     msg=f"Chat-{j} Tentativo {i // 2 + 1}" if i % 2 == 0 else f"Chat-{j} Risposta {i // 2 + 1}",
#                     score=(torch.rand(1) * 10).item(),
#                 ) for i in range(10)
#             ]
#             for j in range(37)
#         ],
#         model_response=
#         """ 
#             Lorem ipsum dolor sit amet, consectetur adipiscing elit. Integer id ante quis mauris commodo feugiat a a sem. Nunc elementum eget arcu ac molestie. Mauris interdum metus id tortor finibus, vitae egestas sapien vestibulum. Ut vel orci et est volutpat commodo id eu dui. Aenean vulputate dapibus ex, cursus tristique arcu egestas sed. Cras molestie aliquam nulla, in convallis nisi tempor vel. Sed ut tempor odio, in tempor ex. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia curae; Curabitur lobortis feugiat mollis. Donec mattis lobortis euismod. Phasellus pellentesque augue non mollis condimentum. Nam non laoreet dolor, sit amet ultricies turpis. Etiam risus lorem, pulvinar quis neque in, pulvinar vulputate augue. Phasellus quis purus sed eros elementum laoreet et sit amet justo. Nam facilisis risus eget tortor pharetra dictum. Nunc fringilla mi ut mi consectetur, dictum ultrices diam pellentesque.
#             Nunc tristique dui molestie nisl viverra, id ornare risus facilisis. Duis bibendum eget ipsum at suscipit. Aenean pharetra a elit eu placerat. Nullam purus felis, tincidunt sit amet viverra ac, laoreet sit amet dui. Interdum et malesuada fames ac ante ipsum primis in faucibus. Sed in leo orci. Nam tincidunt tempus lacus vel commodo. Curabitur volutpat diam ac orci mattis hendrerit. Phasellus vitae pellentesque arcu. Sed ornare interdum turpis, vitae ultrices est auctor et. Sed vitae erat efficitur, volutpat dui sed, ornare massa. Proin viverra libero vitae tincidunt ultricies. Duis cursus tempor purus, vitae porttitor dui condimentum sed. Suspendisse id arcu sit amet nulla sodales ultrices sed et arcu. Donec tempor cursus vulputate.
#             Maecenas quis bibendum ligula, nec varius quam. Maecenas tortor risus, molestie sit amet eros ut, faucibus venenatis urna. Vivamus ut diam quis mi faucibus molestie non sed urna. Etiam tempus est lectus, eu faucibus odio fringilla eget. Donec nulla erat, hendrerit sit amet dapibus in, bibendum a nulla. Phasellus tincidunt ipsum lectus. Cras quis mattis quam. Mauris at libero egestas, auctor magna ut, ornare diam. Integer et risus auctor, blandit nisl consequat, venenatis dui. Aliquam eget congue neque.
#             Class aptent taciti sociosqu ad litora torquent per conubia nostra, per inceptos himenaeos. Proin tortor dui, finibus quis efficitur eget, ultricies id massa. Fusce odio nisl, varius non eros non, iaculis fermentum enim. Sed pulvinar augue in erat porta, quis dignissim arcu lobortis. Maecenas a congue neque. Donec accumsan, magna vitae posuere convallis, orci sapien porttitor risus, vel porttitor nunc libero a ex. Suspendisse accumsan egestas venenatis. Vivamus eu lectus congue, pharetra ex sed, euismod ex. Class aptent taciti sociosqu ad litora torquent per conubia nostra, per inceptos himenaeos. Duis tempus ullamcorper ante, eu convallis magna accumsan quis. Phasellus lacinia a nisi non ultrices. Integer in velit sed tellus elementum consectetur quis eget leo. Nam molestie in augue iaculis suscipit. Aenean ullamcorper eros in tempus accumsan.
#             Nam vulputate, magna vel hendrerit auctor, magna arcu ornare mauris, vel fermentum risus eros quis purus. Sed hendrerit fermentum eleifend. Proin ut turpis venenatis, posuere nisl ut, vehicula elit. Phasellus tellus nibh, pulvinar ac odio in, efficitur suscipit leo. Donec quis ante eros. Aliquam at justo felis. Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas. Aliquam consequat sodales tristique. Vivamus pulvinar, eros vel sollicitudin finibus, est diam bibendum nisl, sit amet volutpat urna felis non diam. 
#         """,
#         advance_metrics={
#             "attack success rate": 0.7
#         }
#     )

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
