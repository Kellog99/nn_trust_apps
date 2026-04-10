import time
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated, Any, Optional, Type, cast

import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
import torchvision
from annotated_types import Ge, Le
from pydantic import Field, NonNegativeFloat
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

from nn_trust import Task, AttackType, Knowledge
from nn_trust.attack import EvasionAttackFactory
from nn_trust.attack._evasion import EvasionAttack, EvasionAttackConfig
from nn_trust.attack.detection_utils import absolutize, nms, xywh2x1y1wh
from nn_trust.attack.normalization import ClampNormalization, Normalization
from nn_trust.attack.transforms import AddUniformNoiseTransform
from nn_trust.attack.utils._utils import to_device
from nn_trust.loss import DetectionLoss, TotalVariationLoss


class _PatchTransform(nn.Module):
    """
    Transformation on the patches for AdvYOLO-
    """

    def __init__(self):
        super().__init__()
        self.cj = transforms.ColorJitter(brightness=(0.8, 1.2),
                                         contrast=(0.8, 1.2))  # , saturation=(0.9, 1.1), hue=(-0.05, 0.05))
        self.noise = AddUniformNoiseTransform(noise_range=(0.0, 0.1))
        self.rotation = transforms.RandomRotation(degrees=(-20, 20), expand=True, fill=-1)

    def forward(self, x):
        x = self.rotation(x)
        mask = (x[0] >= 0) * 1.0

        x = self.cj(x)
        x = self.noise(x)

        return x, mask


_BG_TRANSFORM = transforms.Compose(
    [
        transforms.ColorJitter(brightness=(0.6, 1.4), contrast=(0.6, 1.4), saturation=(0.6, 1.4), hue=(-0.1, 0.1)),
        transforms.RandomResizedCrop(size=[640, 640], scale=(0.5, 1.0)),
    ]
)


class AdvYoloAttackConfig(EvasionAttackConfig):
    # Patch optimization params
    optimizer: Type[optim.Optimizer] = Field(default=optim.Adam, description="Patch optimizer.")
    optimizer_params: dict[str, Any] = Field(
        default_factory=lambda: dict(lr=1e-03, weight_decay=1e-04),
        description="Parameters of the optimizer, the learning rate 'lr' is the minimum parameter required.",
    )
    lr_scheduler: Optional[Type[optim.lr_scheduler.LRScheduler]] = Field(
        default=None, description="If provided, the learning rate scheduler."
    )
    lr_scheduler_params: dict[str, Any] = Field(
        default_factory=lambda: dict(), description="The parameters of the learning rate scheduler."
    )
    save: bool = Field(default=True, description="If True, the last patch and the best will be saved.")
    patch_size: int = Field(
        default=200,
        description="Patch dimension.",
        gt=1,
        title="Path Size")

    # Patch normalization
    patch_normalizer: Normalization = Field(
        default=ClampNormalization(0, 1), description="The normalization applied to the patch."
    )

    # Augmentation params
    background_transforms: Optional[nn.Module] = Field(
        default=lambda x: x, description="The transformation to apply to augment the background image."
    )
    patch_transforms: Optional[nn.Module] = Field(
        default=lambda x: x, description="The transformation function to apply to the patch before applying.")

    patch_resize_range: tuple[float, float] = Field(
        default=(0.1, 0.2),
        description="Resize factor w.r.t. the length of the boxes diagonal."
    )
    patch_apply_range: tuple[tuple[float, float], tuple[float, float]] = Field(
        default=((0.3, 0.7), (0.33, 0.66)),
        description="Normalized range where to apply the patch."
    )

    # Detection params
    iou_threshold: float = Field(
        default=0.5,
        description="The IOU threshold to find bboxes to apply the patches.",
        ge=0.0,
        le=1.0,
        title="Threshold IOU"
    )
    score_threshold: float = Field(
        default=0.5,
        description="The object score threshold to find bboxes to apply the patches.",
        ge=0.0,
        le=1.0,
        title="Threshold score"
    )

    # Detection params for loss
    iou_threshold_loss: float = Field(
        default=0.1,
        description="The IOU threshold to find bboxes to compute the loss.",
        ge=0.0,
        le=1.0,
        title="Threshold Loss IOU"
    )

    score_threshold_loss: float = Field(
        default=0.1,
        description="The object score threshold to find bboxes to compute the loss.",
        ge=0.0,
        le=1.0,
        title="Threshold Loss for the BBox"
    )

    # The classes of the boxes to attack (to remove)
    label_target: int = Field(
        default_factory=lambda: 0,
        description="The classes to be attacked in the model.",
        ge=0,
        title="Target Label id"
    )

    # Loss params
    total_variation_weight: float = Field(
        default=0.5,
        description="The TV loss contribution.",
        gt=0.0,
        title="Total Variation weight"
    )


@EvasionAttackFactory.register(
    name="Adversarial YOLO",
    description="This attack generates patch that can be physically generated to attack YOLO network.",
    task={Task.Detection},
    type=AttackType.Physical,
    knowledge=Knowledge.White
)
class AdvYoloAttack(EvasionAttack):
    """
    Implements the AdvYOLO patch. It needs the class to attack to generate a patch for that specific class.
    Here the NPS is not used.

    S. Thys, et al. "Fooling automated surveillance cameras: adversarial patches to attack person detection," in CoRR,
    vol. abs/1904.08653, 2019.
    """

    CONFIG_T = AdvYoloAttackConfig

    def __init__(self, config: EvasionAttackConfig) -> None:
        super().__init__(config)
        self._config = cast(AdvYoloAttackConfig, self._config)

        # Copy of model onto the same device
        self._config.model.to(self._config.device)

        # Initialize patch, optimizer, and lr scheduler
        self.patch = torch.rand((3, self._config.patch_size, self._config.patch_size)).to(self._config.device)
        self.patch.requires_grad_(True)
        self.optim = self._config.optimizer([self.patch], **self._config.optimizer_params)
        if self._config.lr_scheduler is not None:
            self.lr_scheduler = self._config.lr_scheduler(self.optim, **self._config.lr_scheduler_params)
        # Stats to get the best patch
        self._best_epoch = 0
        self._best_loss = 1.0
        self.best_patch = self.patch.detach().clone()

        # Create loss objects
        self._criterion_det = DetectionLoss(
            combine="sum", reduction="mean", weight_objectiveness=1.0, weight_classes=0.0
        )
        self._criterion_tv = TotalVariationLoss()

    def generate(
            self,
            x: Optional[torch.tensor] = None,
            y: Optional[torch.tensor] = None,
            ext_results: Optional[dict] = None,
            gen_train: Optional[Iterator[tuple[torch.Tensor, Optional[torch.Tensor]]]] = None,
            gen_val: Optional[Iterator[tuple[torch.Tensor, Optional[dict[str, torch.Tensor]]]]] = None,
            **kwargs,
    ) -> torch.tensor:
        """
        :param x: The images' data.
        :param y: The bbox labels data in detection format.
        :param ext_results: A dict where to save more results.
        :param gen_train: A generator of x, Optional[y] data to use batches instead of one single tensor of data.
        :param gen_val: A generator of x, Optional[y] data to also validate the patch using mAP metric on validation.

        :return: The adversarial patch.
        """
        # Create a folder for the current run
        run_path = Path("AdvYOLO_runs", time.strftime("%Y%m%d-%H%M%S"))
        run_path.mkdir(parents=True, exist_ok=True)
        writer_train = SummaryWriter(str(run_path / "train"))
        writer_val = SummaryWriter(str(run_path / "val"))

        # Instance the loop on epochs
        epoch_loop = range(self._best_epoch, self._best_epoch + self._config.max_iters)
        if self._config.verbose:
            epoch_loop = tqdm(epoch_loop, initial=self._best_epoch, desc="Generating patch ...", position=0)

        for i in epoch_loop:
            # Select between single batch and generator
            if x is not None:
                data_loop = [(x, y)]
            else:
                data_loop = gen_train
            if self._config.verbose:
                data_loop = tqdm(data_loop, desc="Training patch ...", leave=False, position=1)

            epoch_loss_det = torch.tensor(0.0, device=self._config.device)
            epoch_loss_tv = torch.tensor(0.0, device=self._config.device)
            epoch_loss = torch.tensor(0.0, device=self._config.device)

            for x_batch, _ in data_loop:
                x_batch = x_batch.to(self._config.device)

                # Apply bg transformations
                x_batch = self._config.background_transforms(x_batch)

                # Get original predictions
                with torch.no_grad():
                    pred_on_orig = self._config.model(x_batch)
                    pred_on_orig = nms(pred_on_orig, self._config.iou_threshold, self._config.score_threshold)

                # Put patch on detection of people and optimize to reduce those
                self.optim.zero_grad()
                adv_batch = torch.stack(
                    [
                        self.apply_patch(
                            x_batch[k], pred_on_orig[k]["boxes"][pred_on_orig[k]["labels"] == self._config.label_target]
                        )
                        for k in range(len(pred_on_orig))
                    ]
                )

                # plt.imshow(adv_batch[0].detach().cpu().permute(1, 2, 0))
                # plt.show()

                # Get patched predictions
                pred_on_adv = self._config.model(adv_batch)
                pred_on_adv = nms(pred_on_adv, self._config.iou_threshold_loss, self._config.score_threshold_loss)

                # Compute loss
                loss = self._loss(pred_on_adv)

                # Stats memory
                epoch_loss_det += loss["loss_det"].detach()
                epoch_loss_tv += loss["loss_tv"].detach()
                epoch_loss += loss["loss"].detach()

                # Optimize patch
                loss["loss"].backward()
                self.optim.step()

                # Normalize patch
                self.patch.data = self._config.patch_normalizer.normalize(self.patch.data)

            # Update tensorboard
            writer_train.add_scalar("Loss/detection", epoch_loss_det / len(data_loop), i)
            writer_train.add_scalar("Loss/total_variation", epoch_loss_tv / len(data_loop), i)
            writer_train.add_scalar("Loss/total", epoch_loss / len(data_loop), i)

            if gen_val is not None:
                # Compute mAP on validation
                map_metric = torchmetrics.detection.MeanAveragePrecision(box_format="cxcywh").to(self._config.device)

                if self._config.verbose:
                    data_loop = tqdm(gen_val, desc="Validating patch ...", leave=False, position=1)

                epoch_loss_det_v = torch.tensor(0.0, device=self._config.device)
                epoch_loss_tv_v = torch.tensor(0.0, device=self._config.device)
                epoch_loss_v = torch.tensor(0.0, device=self._config.device)

                for x_batch, y_batch in data_loop:
                    x_batch = x_batch.to(self._config.device)
                    y_batch = to_device(y_batch, self._config.device)

                    with torch.no_grad():
                        # Predict in validation data
                        pred_orig = self._config.model(x_batch)
                        pred_orig = nms(pred_orig, self._config.iou_threshold, self._config.score_threshold)
                        # Apply patches
                        adv_batch = torch.stack(
                            [
                                self.apply_patch(
                                    x_batch[k],
                                    pred_orig[k]["boxes"][pred_orig[k]["labels"] == self._config.label_target],
                                )
                                for k in range(len(pred_orig))
                            ]
                        )
                        # Predict on patched data
                        pred_adv = self._config.model(adv_batch)
                        pred_adv = nms(pred_adv, self._config.iou_threshold_loss, self._config.score_threshold_loss)

                    loss = self._loss(pred_adv)
                    map_metric.update(pred_adv, y_batch)

                    epoch_loss_det_v += loss["loss_det"]
                    epoch_loss_tv_v += loss["loss_tv"]
                    epoch_loss_v += loss["loss"]

                map_val = map_metric.compute()

                epoch_loss_det_v /= len(data_loop)
                epoch_loss_tv_v /= len(data_loop)
                epoch_loss_v /= len(data_loop)

                # Update tensorboard
                writer_val.add_scalar("Loss/detection", epoch_loss_det_v, i)
                writer_val.add_scalar("Loss/total_variation", epoch_loss_tv_v, i)
                writer_val.add_scalar("Loss/total", epoch_loss_v, i)
                writer_val.add_scalar("mAP/map", map_val["map"], i)
                writer_val.add_scalar("mAP/map_50", map_val["map_50"], i)
                writer_val.add_scalar("mAP/map_75", map_val["map_75"], i)

                # Save last patch
                if self._config.save:
                    self.save_checkpoint(run_path / "last.pt", i, epoch_loss_v.item())
                # Save best patch if mAP is lower
                if self._best_loss > epoch_loss_v.item():
                    self._best_loss = epoch_loss_v.item()
                    self.best_patch = self.patch.clone().detach()
                    if self._config.save:
                        self.save_checkpoint(run_path / "best.pt", i, epoch_loss_v.item())

                # Save image of the patch
                torchvision.utils.save_image(self.patch.cpu().detach(), run_path / f"{i}.jpg")
                writer_train.add_image("Patch", self.patch.cpu().detach(), i)

                # Update tqdm bar
                if self._config.verbose:
                    epoch_loop.set_postfix(
                        dict(
                            map=map_val["map"].item(),
                            map50=map_val["map_50"].item(),
                            map75=map_val["map_75"].item(),
                            best_loss=self._best_loss,
                        )
                    )

            # Step the lr_scheduler each epoch
            if self._config.lr_scheduler is not None:
                self.lr_scheduler.step()

        return self.best_patch.cpu()

    def _loss(self, pred: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        #  -- detection loss mediated on batch
        loss_det = torch.tensor(0.0, device=self._config.device)
        for j in range(len(pred)):
            idxs = pred[j]["labels"] == self._config.label_target
            cls_label = torch.tensor(
                [self._config.label_target] * torch.nonzero(idxs).size(0), dtype=torch.int32, device=self._config.device
            )  # Select target class
            loss_det += self._criterion_det(pred[j]["scores"][idxs], pred[j]["cls_scores"][idxs], cls_label)
        loss_det /= len(pred)
        #  -- compute tv on patch
        loss_tv = self._criterion_tv(self.patch.unsqueeze(0))
        #  -- total loss
        loss = loss_det + self._config.total_variation_weight * loss_tv
        return dict(loss=loss, loss_tv=loss_tv, loss_det=loss_det)

    def save_checkpoint(self, path: str | Path, i: int, loss: float) -> None:
        torch.save(
            {
                "patch": self.patch,
                "optim_state_dict": self.optim.state_dict(),
                "lr_scheduler_state_dict": self.lr_scheduler.state_dict()
                if self._config.lr_scheduler is not None
                else None,
                "it": i,
                "loss": loss,
            },
            path,
        )

    def load_checkpoint(
            self,
            path: str | Path,
    ) -> None:
        checkpoint = torch.load(path, map_location=self._config.device)
        self._best_epoch = checkpoint["it"] + 1
        self._best_loss = checkpoint["loss"]
        self.patch = checkpoint["patch"].to(self._config.device)
        self.optim = self._config.optimizer([self.patch])
        self.optim.load_state_dict(checkpoint["optim_state_dict"])
        self.lr_scheduler = self._config.lr_scheduler(self.optim, **self._config.lr_scheduler_params)
        if checkpoint["lr_scheduler_state_dict"] is not None:
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])

    def apply_patch(
            self,
            img: torch.Tensor,
            boxes: torch.Tensor,
    ) -> torch.Tensor:
        assert len(img.size()) == 3, "Expected 3D tensor (C * H * W)."
        for box in absolutize(xywh2x1y1wh(boxes), img.size(2), img.size(1)):
            # Generate a patch
            patch, mask = self._config.patch_transforms(self.patch)

            # Resize patch for the bbox dimension
            x1, y1, w, h = box

            # Compute patch dimension as a random portion of the diagonal length of the box
            lo, hi = self._config.patch_resize_range
            ratio = lo + torch.rand(1, device=self._config.device) * (hi - lo)
            p_size = max(1, int(ratio * torch.sqrt(w ** 2 + h ** 2)))

            # Now resize the patch and its mask
            patch = nn.functional.interpolate(
                patch.unsqueeze(0), size=[p_size, p_size], mode="bilinear", align_corners=False
            ).squeeze(0)
            mask = (
                nn.functional.interpolate(mask.unsqueeze(0).unsqueeze(0), size=[p_size, p_size], mode="nearest")
                .squeeze(0)
                .squeeze(0)
            )

            # Attach on the original image
            x_lo, x_hi = self._config.patch_apply_range[0]
            y_lo, y_hi = self._config.patch_apply_range[1]
            p_x = int(x1 + ((x_lo + torch.rand(1, device=self._config.device) * (x_hi - x_lo)) * w))
            p_y = int(y1 + ((y_lo + torch.rand(1, device=self._config.device) * (y_hi - y_lo)) * h))
            offset_l = p_size // 2
            offset_r = p_size - offset_l
            # Calculate the boundaries for slicing
            y_start = max(0, p_y - offset_l)
            y_end = min(p_y + offset_r, img.size(1) - 1)
            x_start = max(0, p_x - offset_l)
            x_end = min(p_x + offset_r, img.size(2) - 1)
            # Calculate the corresponding slice for res_patch and res_mask
            patch_y_start = max(0, -(p_y - offset_l))
            patch_y_end = p_size - max(0, p_y + offset_r - (img.size(1) - 1))
            patch_x_start = max(0, -(p_x - offset_l))
            patch_x_end = p_size - max(0, p_x + offset_r - (img.size(2) - 1))
            # Mask to condition
            mask = mask > 0.5  # 0.5 because of the interpolation
            # Slice the original image and apply the mask
            img[:, y_start:y_end, x_start:x_end] = torch.where(
                mask[patch_y_start:patch_y_end, patch_x_start:patch_x_end],
                patch[:, patch_y_start:patch_y_end, patch_x_start:patch_x_end],
                img[:, y_start:y_end, x_start:x_end],
            )
        return img

    def __repr__(self):
        return "Adversary YOLO"
