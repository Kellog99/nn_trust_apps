import base64
import io
import json
import logging
import os
import time
from pathlib import Path

import timm
import torch
from PIL import Image
from fastapi import APIRouter
from fastapi import Body
from nn_trust.attack.attack_factory import EvasionAttackFactory as EAF
from nn_trust.core import Task, ModelAdapter
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchvision import transforms

from lib.model import SingleAttackOutput, SingleAttackProps, ModelInfo, Error

router = APIRouter(prefix="/job", tags=["jobs management", "jobs utils"])


# --- Single attack --- #
@router.post("/attack")
async def startSingleAttack(body: SingleAttackProps = Body(...)) -> SingleAttackOutput | Error:
    """
    Start a new TITANN attack on single image job.
    """
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ###################### Extracting the Model ######################
        model: ModelAdapter | None = None
        model_info: ModelInfo | None = None
        # Folder where all the models are stored.
        models_root_dir = Path(os.environ.get("INTERNAL_MODEL_STORAGE"))
        for item in models_root_dir.iterdir():
            file_path = models_root_dir / item / "info.json"
            model_path = models_root_dir / item / "model.pth"
            with open(file_path, 'r') as json_file:
                json_file = json.load(json_file)
            model_info = ModelInfo(**json_file)

            if model_info.id == body.id_model:
                # At this moment the assumption is that if there is not a model
                # then it is a Timm model
                if model_path.exists():
                    # check whether it exists a Pytorch model
                    with open(json_file, "r") as f:
                        tmp = torch.load(model_path)
                else:
                    tmp = timm.create_model(model_info.id, pretrained=True)
                model = ModelAdapter(tmp, task=Task.Classification).to(device)
                model.eval()
                break

        if model is None or model_info is None:
            return Error(code=404,
                         message=f"The model {body.id_model} has not been found")
        ###################### Input ######################
        # Decode base64 image string and convert to torch tensor
        image_bytes = base64.b64decode(body.image)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        transformation = transforms.Compose([
            transforms.Resize(model_info.input_dimensionality[1:]),  # Resize BEFORE converting to tensor
            transforms.ToTensor(),  # Convert to tensor AFTER resizing
        ])

        x: torch.Tensor = transformation(image).unsqueeze(0).to(device)
        labels = model(x).argmax(1)
        y: torch.Tensor = torch.nn.functional.one_hot(labels, num_classes=1000)
        ###################### Attack ######################
        parameters = {param.id: param.default for param in body.attack.parameters}
        atk_cnf = EAF.get_config(
            class_id=body.attack.id,
            model=model,
            task=Task.Classification,
            device=device,
            **parameters
        )

        atk = EAF.create(
            body.attack.id,
            atk_cnf
        )

        start = time.time()
        x_adv = atk.generate(x=x, y=y).detach()
        pert = x_adv - x
        end = time.time()
        y_adv = model(x_adv).argmax(-1)

        ###################### Analysing the results ######################
        # Prepare image data to return
        buffered = io.BytesIO()
        transforms.ToPILImage()(x_adv[0]).save(buffered, format="PNG")
        adv_img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        buffered = io.BytesIO()
        pert_image = transforms.ToPILImage()(pert[0])
        pert_image.save(buffered, format="PNG")
        pert_image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        # Execute the attack and get results
        ssim = StructuralSimilarityIndexMeasure().to(device)

        return SingleAttackOutput(
            x=body.image,
            adv_perturbation=pert_image_base64,
            x_adv=adv_img_base64,
            original_prediction=str(y.argmax(-1).item()),
            adversarial_prediction=str(y_adv.item()),
            confidence={
                "adversarial": [],
                "original": []
            },
            advance_metrics={
                "ssim": ssim(x, x_adv).item(),
                "distance": torch.norm(pert, p=1).item(),
                "executionTime": end - start
            })

    except Exception as e:
        logging.error(f"Unexpected error during attack: {str(e)}")

        return Error(
            code=500,
            message=f"Unexpected error during attack"
        )
