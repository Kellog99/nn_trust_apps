import datetime
from pathlib import Path

import time
import torch
from PIL import Image
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchvision.transforms import v2 as T, InterpolationMode

from models import SingleAttackOutput
from nn_trust import Task, CVModelAdapter, EvasionAttack
from nn_trust.models.ultralytics_models import UltralyticsCVModel
from nn_trust.utils.logger import PyTorchCheckpointLogger
from services.utils.utils import tensor_image_to_b64str, draw_predictions
from models.info import Transformation
from nn_trust.attack.utils.detection import nms


def single_attack_performance(
        model: CVModelAdapter,
        attack: EvasionAttack,
        pil_image: Image.Image,
        task: Task,
        input_dimensionality: list[int] | int = [224, 224],
        transformation: Transformation | None = None,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
) -> SingleAttackOutput:
    ################## IMAGE ##################
    if isinstance(input_dimensionality, list):
        if len(input_dimensionality) == 3:
            input_dimensionality = input_dimensionality[1:]
        elif len(input_dimensionality) == 1:
            input_dimensionality = [input_dimensionality[0], input_dimensionality[0]]
        input_dimensionality = tuple(input_dimensionality)
    elif isinstance(input_dimensionality, int):
        input_dimensionality = (input_dimensionality, input_dimensionality)

    # Force a fixed, model-expected channel count (fixes silent C-mismatch -> CUDA assert)
    pil_image = pil_image.convert("RGB")

    ############ image transformation ############
    original_input: torch.Tensor = T.ToTensor()(pil_image)
    print(original_input.shape)
    C, H, W = original_input.shape

    mean = transformation.mean if transformation is not None else [0.485, 0.456, 0.406]
    std = transformation.std if transformation is not None else [0.229, 0.224, 0.225]

    transformations = T.Compose([
        T.Resize(size=input_dimensionality),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=mean, std=std)
    ])

    inv_transform = T.Compose([
        T.Normalize(
            mean=[-m / s for m, s in zip(mean, std)],
            std=[1 / s for s in std]
        ),
        T.Resize(size=(H, W), interpolation=InterpolationMode.BICUBIC),
    ])

    x: torch.Tensor = transformations(pil_image)
    if x.dim() == 3:
        x = x.unsqueeze(0)
    x = x.to(device)
    if not torch.isfinite(x).all():
        raise ValueError("Image preprocessing produced non-finite values.")
    print(" Image loaded ".center(40, "#"))
    ###############################################

    ################## Results ##################
    with torch.no_grad():
        out = model(x)

    out_path = Path(f"./tmp/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    logger: PyTorchCheckpointLogger = PyTorchCheckpointLogger(
        states=["conf_adversarial", "conf_original"],
        path=out_path
    )
    
    match task:
        case Task.Classification:
            #if not torch.isfinite(y).all():
            if not torch.isfinite(out).all():    
                raise RuntimeError(
                    "The model produced non-finite logits for the original image. "
                    "Check the model weights and preprocessing configuration."
                )
            labels = out.argmax(-1).tolist()
            y_pred = str(labels[0])

            print(" Executing the Attack ".center(40, "#"))
            start = time.time()
            x_adv = attack.generate(
                x=x,
                y=out.detach(),
                logger=logger
            ).detach()
            end = time.time()
            if not torch.isfinite(x_adv).all():
                raise RuntimeError(
                    "The attack produced non-finite values. Reduce the attack epsilon/learning rate "
                    "or enable epsilon-ball projection."
                )

            with torch.no_grad():
                print(" Attack Completed ".center(40, "#"))
                adv_logits = model(x_adv)
                if not torch.isfinite(adv_logits).all():
                    raise RuntimeError(
                        "The model produced non-finite logits for the adversarial image. "
                        "Reduce the attack epsilon/learning rate or enable epsilon-ball projection."
                    )
                y_adv: torch.Tensor = adv_logits.argmax(-1)

            y_pred_adv = str(y_adv.item())

        case Task.Detection:
            if not isinstance(model, UltralyticsCVModel):
                raise ValueError("The model must be an instance of UltralyticsCVModel for detection tasks.")

            boxes, scores = out

            if boxes.ndim != 3 or boxes.shape[-1] != 4 or scores.ndim != 3:
                raise ValueError(
                    "Expected YOLO-style detection output: boxes [B, N, 4], scores [B, N, C]."
                )

            iou_threshold = attack.config.iou_threshold_targeted
            score_threshold = attack.config.score_threshold_targeted

            print(" Executing the Attack ".center(40, "#"))
            start = time.time()

            x_adv = attack.generate(
                x=x,
                y=out,
                logger=logger,
            ).detach()

            end = time.time()

            if not torch.isfinite(x_adv).all():
                raise RuntimeError(
                    "The attack produced non-finite values. Reduce the attack epsilon/learning rate "
                    "or enable epsilon-ball projection."
                )

            with torch.no_grad():
                print(" Attack Completed ".center(40, "#"))
                out_adv = model(x_adv)

            adv_boxes, adv_scores = out_adv

            if not torch.isfinite(adv_boxes).all() or not torch.isfinite(adv_scores).all():
                raise RuntimeError(
                    "The model produced non-finite detection outputs for the adversarial image."
                )

            post_nms_preds = nms(
                {
                    "boxes": boxes,
                    "scores": scores.max(dim=-1).values,
                    "cls_scores": scores,
                },
                iou_threshold=iou_threshold,
                score_threshold=score_threshold,
            )

            post_nms_preds_adv = nms(
                {
                    "boxes": adv_boxes,
                    "scores": adv_scores.max(dim=-1).values,
                    "cls_scores": adv_scores,
                },
                iou_threshold=iou_threshold,
                score_threshold=score_threshold,
            )

            x_vis = inv_transform(x.cpu())[0].clamp(0, 1)
            x_adv_vis = inv_transform(x_adv.cpu())[0].clamp(0, 1)

            x_with_pred = draw_predictions(x_vis, post_nms_preds[0])
            x_adv_with_pred = draw_predictions(x_adv_vis, post_nms_preds_adv[0])

            y_pred = tensor_image_to_b64str(x_with_pred.float() / 255)
            y_pred_adv = tensor_image_to_b64str(x_adv_with_pred.float() / 255)

        case _:
            raise ValueError(f"Unsupported task: {task}")
            
    ssim_metric = StructuralSimilarityIndexMeasure().to(device)
    ssim_measure: float = ssim_metric(x.to(device), x_adv.to(device)).item()

    conf_original: list[list[float]] = logger.get_log(tag="conf_original")
    conf_adversarial: list[list[float]] = logger.get_log(tag="conf_adversarial")

    conf_original: list[float] = [conf[0] for conf in conf_original]
    conf_adversarial: list[float] = [conf[0] for conf in conf_adversarial]
    ################## Invert transform ################
    pert: torch.Tensor = x_adv.cpu() - x.cpu()
    # A perturbation has no mean component: inverse-normalize it by scaling
    # with the channel standard deviation only. Applying Normalize here would
    # add the ImageNet mean and inflate the reported distance.
    std_tensor = torch.tensor(std, dtype=pert.dtype).view(1, 3, 1, 1)
    pert = T.Resize(size=(H, W), interpolation=InterpolationMode.BICUBIC)(pert * std_tensor)
    x_adv = inv_transform(x_adv.cpu())

    return SingleAttackOutput(
        x_adv=tensor_image_to_b64str(x_adv.cpu()),
        adv_perturbation=tensor_image_to_b64str(pert.cpu()),
        original_prediction=y_pred,
        adversarial_prediction=y_pred_adv,
        advance_metrics={
            "ssim": ssim_measure,
            "distance": torch.norm(pert, p=getattr(attack.config, "p", 2)).item(),
            "execution_time": end - start,
        },
        confidence={
            "original": {i: conf for i, conf in enumerate(conf_original)},
            "adversarial": {i: conf for i, conf in enumerate(conf_adversarial)},
        }
    )
