import datetime
import logging
import time
from pathlib import Path

import torch
from PIL import Image
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchvision.transforms import v2 as T, InterpolationMode

from models import SingleAttackOutput
from models.info import Transformation
from nn_trust import Task, CVModelAdapter, EvasionAttack
from nn_trust.attack.utils.detection import nms, LetterboxCocoTransform
from nn_trust.models.ultralytics_models import UltralyticsCVModel
from nn_trust.utils.logger import PyTorchCheckpointLogger
from services.utils.utils import tensor_image_to_b64str, draw_predictions
from utils.dataset_utils import get_transform_dataset, get_inverse_transform

logger = logging.getLogger(__name__)


def _check_finite(x: torch.Tensor, msg: str) -> None:
    if not torch.isfinite(x).all():
        raise RuntimeError(msg)


def _run_attack(
        attack: EvasionAttack,
        x: torch.Tensor,
        y,
        atk_logger: PyTorchCheckpointLogger
) -> tuple[
    torch.Tensor, float]:
    logger.info("Executing attack")
    start = time.time()
    x_adv = attack.generate(
        x=x,
        y=y,
        logger=atk_logger
    ).detach()
    elapsed = time.time() - start
    _check_finite(x_adv, "Attack produced non-finite values; reduce epsilon/learning rate or enable projection.")
    logger.info("Attack completed in %.3fs", elapsed)
    return x_adv, elapsed


def single_attack_performance(
        model: CVModelAdapter,
        attack: EvasionAttack,
        pil_image: Image.Image,
        task: Task,
        input_dimensionality: list[int] | int = (224, 224),
        transformation: Transformation | None = None,
        device: torch.device | None = None,
) -> SingleAttackOutput:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(input_dimensionality, int):
        input_dimensionality = (input_dimensionality, input_dimensionality)
    else:
        if len(input_dimensionality) == 3:
            input_dimensionality = input_dimensionality[1:]
        elif len(input_dimensionality) == 1:
            input_dimensionality = [input_dimensionality[0], input_dimensionality[0]]
        input_dimensionality = tuple(input_dimensionality)

    pil_image = pil_image.convert("RGB")
    H, W = pil_image.height, pil_image.width

    ############ image transformation ############
    match task:
        case Task.Classification:
            transformations = get_transform_dataset(transformation=transformation)
            inv_transform = get_inverse_transform(transformation=transformation, H=H, W=W)

        case Task.Detection:
            if not isinstance(model, UltralyticsCVModel):
                raise ValueError("The model must be an instance of UltralyticsCVModel for detection tasks.")

            letterbox = LetterboxCocoTransform(cat_id_to_label={}, new_shape=input_dimensionality)
            params = letterbox.get_params(image_width=W, image_height=H)

            def transformations(image):
                image, _ = letterbox(image, [])
                return image

            def inv_transform(image: torch.Tensor) -> torch.Tensor:
                cropped = image[
                    :,
                    params["top"]: params["top"] + params["resized_height"],
                    params["left"]: params["left"] + params["resized_width"],
                ]
                return T.Resize(size=(H, W), interpolation=InterpolationMode.BILINEAR)(cropped)

        case _:
            raise ValueError(f"Unsupported task: {task}")

    x: torch.Tensor = transformations(pil_image)
    if x.dim() == 3:
        x = x.unsqueeze(0)
    x = x.to(device)
    _check_finite(x, "Image preprocessing produced non-finite values.")
    logger.info("Image loaded")

    ################## Results ##################
    with torch.no_grad():
        out = model(x)

    out_path = Path(f"./tmp/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}")
    atk_logger = PyTorchCheckpointLogger(states=["conf_adversarial", "conf_original"], path=out_path)

    try:
        match task:
            case Task.Classification:
                _check_finite(out, "Model produced non-finite logits for the original image.")
                y_pred = str(out.argmax(-1).item())

                x_adv, elapsed = _run_attack(attack, x, out.detach(), atk_logger)

                with torch.no_grad():
                    adv_logits = model(x_adv)
                _check_finite(adv_logits, "Model produced non-finite logits for the adversarial image.")
                y_pred_adv = str(adv_logits.argmax(-1).item())

            case Task.Detection:
                boxes, scores = out
                if boxes.ndim != 3 or boxes.shape[-1] != 4 or scores.ndim != 3:
                    raise ValueError("Expected YOLO-style detection output: boxes [B, N, 4], scores [B, N, C].")

                x_adv, elapsed = _run_attack(attack, x, out, atk_logger)

                with torch.no_grad():
                    adv_boxes, adv_scores = model(x_adv)
                _check_finite(adv_boxes, "Model produced non-finite detection boxes for the adversarial image.")
                _check_finite(adv_scores, "Model produced non-finite detection scores for the adversarial image.")

                iou_threshold = attack.config.iou_threshold_evaluation
                score_threshold = attack.config.score_threshold_evaluation

                preds = nms(
                    y={
                        "boxes": boxes,
                        "scores": scores.max(dim=-1).values,
                        "cls_scores": scores
                    },
                    iou_threshold=iou_threshold,
                    score_threshold=score_threshold,
                )
                preds_adv = nms(
                    y={
                        "boxes": adv_boxes,
                        "scores": adv_scores.max(dim=-1).values,
                        "cls_scores": adv_scores
                    },
                    iou_threshold=iou_threshold,
                    score_threshold=score_threshold,
                )

                class_names = model.model.names
                x_drawn = draw_predictions(
                    image=x[0],
                    pred=preds[0],
                    display_top_k=attack.config.display_top_k,
                    class_names=class_names
                )
                x_adv_drawn = draw_predictions(
                    image=x_adv[0],
                    pred=preds_adv[0],
                    display_top_k=attack.config.display_top_k,
                    class_names=class_names
                )

                y_pred = tensor_image_to_b64str(inv_transform(x_drawn).float() / 255)
                y_pred_adv = tensor_image_to_b64str(inv_transform(x_adv_drawn).float() / 255)

            case _:
                raise ValueError(f"Unsupported task: {task}")

        conf_original = [c[0] for c in atk_logger.get_log(tag="conf_original")]
        conf_adversarial = [c[0] for c in atk_logger.get_log(tag="conf_adversarial")]
    finally:
        atk_logger.close()

    ssim_metric = StructuralSimilarityIndexMeasure().to(device)
    ssim_measure = ssim_metric(x, x_adv).item()

    ################## Invert transform ################
    pert: torch.Tensor = x_adv.cpu() - x.cpu()

    if task == Task.Classification:
        std = transformation.std if transformation is not None else [1.0] * pert.shape[1]
        std_tensor = torch.tensor(std, dtype=pert.dtype).view(1, -1, 1, 1)
        pert = T.Resize(size=(H, W))(pert * std_tensor)
        x_adv_output = inv_transform(x_adv.cpu())
    else:
        # Detection tensors are already in YOLO letterboxed [0, 1] space.
        x_adv_output = x_adv.cpu()

    return SingleAttackOutput(
        x_adv=tensor_image_to_b64str(x_adv_output),
        adv_perturbation=tensor_image_to_b64str(pert),
        original_prediction=y_pred,
        adversarial_prediction=y_pred_adv,
        advance_metrics={
            "ssim": ssim_measure,
            "distance": torch.norm(pert, p=getattr(attack.config, "p", 2)).item(),
            "execution_time": elapsed,
        },
        confidence={
            "original": dict(enumerate(conf_original)),
            "adversarial": dict(enumerate(conf_adversarial)),
        },
    )
