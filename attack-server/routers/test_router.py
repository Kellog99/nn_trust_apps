import torch
from fastapi import APIRouter
from fastapi import Body, Query
from torchvision.transforms import v2 as T

from lib.model import RegisteredObject
from models.attack import SingleAttackOutput, SingleAttackProps
from models.info import ModelInfo
from nn_trust import Task
from nn_trust.attack import EvasionAttack
from nn_trust.attack.attack_factory import EvasionAttackFactory as EAF
from utils.attack import single_image_attack
from utils.model import load_model
from utils.utils import b64str_to_pil

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

    Returns:
        SingleAttackOutput: a collection of all the results concerning a single attack.
    """
    ################## MODEL ##################
    model_info: ModelInfo = body.model
    model = load_model(
        model_type=model_info.model_type,
        model_path=model_info.repository,
        task=Task.from_str(model_info.task),
        model_api=model_info.api,
        model_id=model_info.id,
    )
    ###########################################

    ################## IMAGE ##################
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

    transformations = T.Compose([
        T.Resize(input_dimensionality),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
    ])
    x = transformations(pil_image)
    ###########################################

    ################## ATTACK ##################
    attack: RegisteredObject = body.attack
    attack: EvasionAttack = EAF.create(
        model=model,
        class_id=attack.id,
        task=Task.from_str(model_info.task),
        **{param.id: param.default for param in attack.parameters}
    )
    ############################################

    return single_image_attack(
        model=model,
        x=x,
        attack=attack,
        device=torch.device(device),
    )
