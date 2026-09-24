import os

import torch
from torch import nn
from pytorch_lightning import LightningModule
from transformers import AutoModelForImageClassification, AutoImageProcessor
from torchmetrics import Accuracy

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import MLFlowLogger

import ray
from ray.tune.search.optuna import OptunaSearch

from torch.utils.data import Dataset, Subset, DataLoader
from torchvision.datasets import ImageFolder
import lightning as L
import torch
from torchvision.transforms import v2 as T
from typing import Literal
from pytorch_lightning.callbacks import Callback
from datetime import datetime
from pytorch_lightning.callbacks import ModelCheckpoint
import mlflow

class TuneReportCallback(Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        # Get the latest validation accuracy
        val_acc = trainer.callback_metrics.get("val_acc")
        if val_acc is not None:
            val_acc_value = val_acc.item() if hasattr(val_acc, 'item') else float(val_acc)
            ray.tune.report({"val_acc":val_acc_value})


class HFImageClassifier(LightningModule):
    def __init__(
        self,
        model_name: str,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        num_labels: int = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        if num_labels is None:
            raise ValueError("num_labels must be specified")
        
        self.model = AutoModelForImageClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
        )
        
        self.train_acc = Accuracy(task="multiclass", num_classes=num_labels)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_labels)

    def forward(self, pixel_values, labels=None):
        return self.model(pixel_values=pixel_values, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        preds = outputs.logits.argmax(dim=-1)
        
        self.train_acc.update(preds, batch["labels"])
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_acc", self.train_acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        preds = outputs.logits.argmax(dim=-1)
        
        self.val_acc.update(preds, batch["labels"])
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", self.val_acc, prog_bar=True, on_step=False, on_epoch=True)


    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

class ImageFolderDataModule(L.LightningDataModule):
    def __init__(
            self, 
            data_dir: str, 
            batch_size: int = 32,
            seed: int = 42,
            train_size: float = 0.3,
            crop_size: int | tuple[int] | None = None
        ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_size = train_size
        self.seed = seed
        self.crop_size = crop_size
        self.num_classes = None

    @staticmethod
    def stratified_dataset_train_val_split(
        dataset: Dataset, 
        train_size: float = 0.8, 
        random_state: int | None = 42
    ):
        num_classes = len(dataset.classes)
        train_indices, val_indices = [], []
        for c in range(num_classes):
            idx = torch.where(torch.tensor(dataset.targets) == c)[0]
            split = int(len(idx) * train_size)
            train_indices.extend(idx[:split])
            val_indices.extend(idx[split:])
        return Subset(dataset, train_indices), Subset(dataset, val_indices)

    @staticmethod
    def get_transforms(
        split: Literal["train", "val"] = "train", 
        crop_size: int | tuple[int, int] | None = None
        ):

        if split == "train":
            transformations = [
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                # Normalize
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                # Color jittering
                T.ColorJitter(brightness=0.01, contrast=0.02, saturation=0.02, hue=0.01),
                # Rotation/flip
                T.RandomHorizontalFlip(),
                T.RandomRotation(3),
            ]
        else:
            transformations = [
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                # Normalize
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]

        # Crops
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        if crop_size is not None:
            transformations.extend([
                T.CenterCrop(crop_size), 
                T.Resize(crop_size)])
        return T.Compose(transformations)

    def setup(self, stage: str):
        self.ds = ImageFolder(root=self.data_dir)
        self.num_classes = len(self.ds.classes)
        self.train_ds, self.val_ds = self.stratified_dataset_train_val_split(
            self.ds, 
            train_size=self.train_size, 
            random_state=self.seed
        )
        self.train_ds.dataset.transform = self.get_transforms(split="train", crop_size=self.crop_size)
        self.val_ds.dataset.transform = self.get_transforms(split="val", crop_size=self.crop_size)

    @staticmethod
    def collate_fn(batch):
        images, labels = zip(*batch)
        images = torch.stack(images)
        labels = torch.tensor(labels)
        return {
            "pixel_values": images,
            "labels": labels
        }

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, collate_fn=self.collate_fn)

def train_model(config):
    run_id = f"model_{config['model_name']}_lr_{config['lr']}_{config['weight_decay']}_bs_{config['batch_size']}"
    logger = MLFlowLogger(
        experiment_name=config["experiment_name"],
        run_name=run_id,
        tracking_uri="/home/papab/codes/nn_trust_apps/train_classificationv2/mlruns"
    )

    dm = ImageFolderDataModule(
        data_dir=config["data_dir"],
        batch_size=config["batch_size"],
    )
    dm.setup(stage="train")

    model = HFImageClassifier(
        model_name=config["model_name"],
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        num_labels=dm.num_classes
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"./lightning_logs/{config['experiment_name']}/{run_id}",  # Where to save checkpoints
        filename="{epoch}-{val_acc:.2f}",  # Checkpoint naming
        monitor="val_acc",  # Metric to monitor
        mode="max",  # "max" for accuracy, "min" for loss
        save_top_k=3,  # Save top 3 checkpoints
        save_last=True,  # Also save last checkpoint
        verbose=True,
    )

    torch.set_float32_matmul_precision('high')

    trainer = Trainer(
        accelerator="gpu",
        devices="auto",
        strategy="ddp",
        precision='32-true',
        max_epochs=config["epochs"],
        logger=logger,
        callbacks=[checkpoint_callback],
        default_root_dir=f"./lightning_logs/{config['experiment_name']}/{run_id}"
    )

    trainer.fit(model, dm.train_dataloader(), dm.val_dataloader())




