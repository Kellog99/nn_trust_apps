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
from nn_trust.attack import EvasionAttack

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
    attack: EvasionAttack = EAF.create(
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

# --- Jailbreaking --- #
@router.post("/jailbreaking")
async def jailbreaking(
        body: dict = Body(...),
        device: str = Query(
            default="cpu",
            description="The device to run the model on.",
            example="cpu"
        )

):
    """
    Handle the POST request for executing a jailbreak attack using PAIR.
    """
    print(yaml.dump(body, default_flow_style=False, sort_keys=False))

    return {
        "adversarial_prompt": "N/A",
        "conversations": "N/A",
        "model_response": "N/A",
        "advance_metrics": "N/A"
    }

