import json

from argparse import ArgumentParser
from pathlib import Path
from typing import Literal, Optional
from collections import Counter
import logging
from datetime import datetime
import os

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torchvision
import torchmetrics
from torchvision.datasets import ImageFolder
from torch.utils.data.sampler import WeightedRandomSampler
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm

try:
    import pyfiglet

    NO_ASCII_ART = False
except ImportError:
    NO_ASCII_ART = True


@torch.no_grad()
def _compute_mean_std_dataset(
    dataloader: DataLoader, method: Literal["subsample", "naive"] = "subsample"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes the mean and std of a given dataloader in the image domain."""
    mean = torch.zeros((3,), dtype=float)
    sum_squares = torch.zeros((3,), dtype=float)
    if method == "naive":
        running_n = 1
        for batch, _ in (loop := tqdm(dataloader)):
            avg_batch = batch.flatten(2).mean(dim=-1)
            for elem in avg_batch.unbind(0):
                disp_elem = elem - mean
                sum_squares += (running_n - 1) / running_n * (disp_elem * disp_elem)
                mean += disp_elem / running_n
                running_n += 1
            loop.set_postfix({"mean": mean, "std": (sum_squares / (running_n - 1)).sqrt()})
        std = (sum_squares / (running_n - 1)).sqrt()
    elif method == "subsample":
        # NOTE: this requires batch size to be large!
        running_batch = 1
        std = torch.zeros((3,), dtype=float)
        for batch, _ in (loop := tqdm(dataloader)):
            batch = batch.flatten(2).permute(1, 0, 2).flatten(1)
            mean += batch.mean(dim=-1)
            std += batch.std(dim=-1)
            running_batch += 1
            loop.set_postfix({"mean": mean / running_batch, "std": std / (running_batch - 1)})

        mean /= running_batch
        std /= running_batch - 1
    else:
        raise ValueError(f"The method '{method}' is not implemented yet.")

    return mean, std


class MilitaryAircraftDataset(Dataset):
    def __init__(self, root_dir: Path, transform=None, target_transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.images = list(root_dir.glob("**/*/*.jpg"))
        # Get only crops that are larger in size than 2**12
        self.images = list([img for img in self.images])
        self.labels = [img.parts[-2] for img in self.images]
        self.id_to_name = {name: i for i, name in enumerate(sorted(set(self.labels)))}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        # Transform a string label to an integer label
        label = self.labels[idx]
        label = self.id_to_name[label]

        image = torchvision.io.decode_image(image_path, mode=torchvision.io.ImageReadMode.RGB).float() / 255.0
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label


class VisDroneCropsDataset(Dataset):
    def __init__(self, root_dir: Path, transform=None, target_transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.images = list(root_dir.glob("**/*/*.png"))
        # Get only crops that are larger in size than 2**12
        self.images = list([img for img in self.images if img.stat().st_size > 2**12])
        self.labels = [int(img.parts[-2]) for img in self.images]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        label = self.labels[idx]

        image = torchvision.io.decode_image(image_path, mode=torchvision.io.ImageReadMode.RGB).float() / 255.0
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label


def DEFAULT_CROP_VISDRONE(crop_size: int | tuple[int, int] | None = None):
    transformations = [
        torchvision.transforms.ToTensor(),
        # Normalize
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    # Crops
    if isinstance(crop_size, int):
        crop_size = (crop_size, crop_size)
    if crop_size is not None:
        transformations.extend([torchvision.transforms.CenterCrop(crop_size), torchvision.transforms.Resize(crop_size)])

    return torchvision.transforms.Compose(transformations)


def DEFAULT_CROP_VISDRONE_AUGMENT(crop_size: int | tuple[int, int] | None = None):
    transformations = [
        torchvision.transforms.ToTensor(),
        # Normalize
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # Color jittering
        torchvision.transforms.ColorJitter(brightness=0.01, contrast=0.02, saturation=0.02, hue=0.01),
        # Rotation/flip
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomRotation(3),
    ]

    # Crops
    if isinstance(crop_size, int):
        crop_size = (crop_size, crop_size)
    if crop_size is not None:
        transformations.extend([torchvision.transforms.CenterCrop(crop_size), torchvision.transforms.Resize(crop_size)])

    return torchvision.transforms.Compose(transformations)

def DEFAULT_CROP_VISDRONE_AUGMENT_VAL(crop_size: int | tuple[int, int] | None = None):
    transformations = [
        torchvision.transforms.ToTensor(),
        # Normalize
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]

    # Crops
    if isinstance(crop_size, int):
        crop_size = (crop_size, crop_size)
    if crop_size is not None:
        transformations.extend([torchvision.transforms.CenterCrop(crop_size), torchvision.transforms.Resize(crop_size)])

    return torchvision.transforms.Compose(transformations)

INVERSE_NORMALIZATION = torchvision.transforms.Compose(
    [
        torchvision.transforms.Normalize(mean=0.0, std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
        torchvision.transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=1.0),
    ]
)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    lr: float,
    epochs: int,
    weight_decay: float = 5e-5,
    warmup: int = 8,
    step_size: int = 10,
    label_smoothing: float = 0.1,
    device: str = "cuda",
    save_every: int = 2,
    temperature: float = 3.0,
    checkpoint_folder_path: Path = Path("./visdrone"),
):
    writer = SummaryWriter(checkpoint_folder_path)
    model = model.to(device)

    # Define the optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    constant_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=warmup)
    reducer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    def_lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[constant_scheduler, reducer_scheduler], milestones=[warmup]
    )

    # Define the loss function
    criterion = nn.CrossEntropyLoss(reduction="mean", label_smoothing=label_smoothing).to(device)
    n_batches = len(train_loader)
    temperature = torch.tensor([temperature], device=device)

    for epoch in (loop := tqdm(range(epochs), ascii=" ▖▘▝▗▚▞█", colour="#3f12f2")):
        cum_loss = 0.0
        acc = 0.0
        n_batches = 1
        cum_val_loss = 0.0
        val_acc = 0.0
        val_n_batches = 1
        model.train()
        for i, (data, target) in enumerate(tqdm(train_loader, leave=False, colour="#1fff21")):
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad()
            # Output with temperature
            output = model(data) / (temperature.log() / (epoch + 1)).exp()
            loss = criterion(output, target)
            loss.backward()
            writer.add_scalar("train/loss", loss.item(), i + epoch * len(train_loader))
            optimizer.step()

            # update stats
            cum_loss += loss.item()
            correct = ((output.argmax(dim=-1) == target) * 1.0).mean()
            acc += correct.item()
            n_batches += 1

        model.eval()
        with torch.no_grad():
            for i, (data, target) in enumerate(tqdm(val_loader, leave=False, colour="#e69123")):
                data = data.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                output = model(data)
                val_loss = criterion(output, target)
                cum_val_loss += val_loss.item()
                correct = ((output.argmax(dim=-1) == target) * 1.0).mean()
                val_acc += correct.item()
                val_n_batches += 1

        def_lr_scheduler.step()
        cum_loss /= n_batches
        acc /= n_batches
        cum_val_loss /= val_n_batches
        val_acc /= val_n_batches
        writer.add_scalar("train/avg_loss", cum_loss, epoch * n_batches)
        writer.add_scalar("train/acc", acc, epoch * n_batches)
        writer.add_scalar("train/lr", def_lr_scheduler.get_last_lr()[0], epoch * n_batches)
        writer.add_scalar("val/avg_loss", cum_val_loss, epoch)
        writer.add_scalar("val/acc", val_acc, epoch)
        loop.set_postfix(
            {"epoch": epoch, "train_loss": cum_loss, "train_acc": acc, "val_loss": cum_val_loss, "val_acc": val_acc}
        )
        if (epoch + 1) % save_every == 0:
            torch.save(model, checkpoint_folder_path / f"model_checkpoint-acc={acc}-epoch={epoch}.pth")

    # Save the final model

    return model


@torch.no_grad()
def test(
    model_checkpoint: Path,
    test_dataloader: DataLoader,
    device: torch.device,
    evaluation_folder_path: Path,
    metrics: list[torchmetrics.Metric] | None = None,
):
    writer = SummaryWriter(evaluation_folder_path)

    model = torch.load(model_checkpoint, weights_only=False)
    model = model.eval()
    model = model.to(device)
    for metric in metrics:
        metrics[metric] = metrics[metric].to(device)
        metrics[metric].reset()

    for data, target in tqdm(test_dataloader, ascii=reversed("▖▘▝▗▚▞█ "), colour="#3f12f2"):
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        prediction = model(data)
        if metrics:
            for metric in metrics:
                if metric == "accuracy":
                    metrics[metric].update(prediction.argmax(dim=-1), target)
                else:
                    metrics[metric].update(prediction, target)

    if metrics:
        for metric in metrics:
            res = metrics[metric].compute()
            if res.dim() > 1:
                writer.add_tensor(f"test/{metric}", res)
            elif res.dim() == 1:
                if res.size(0) > 1:
                    writer.add_tensor(f"test/{metric}", res)
                else:
                    writer.add_scalar(f"test/{metric}", res.item())
            else:
                writer.add_scalar(f"test/{metric}", res.item())
    writer.close()


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


def stratified_dataset_train_val_split(dataset: Dataset, train_size: float = 0.8, random_state: Optional[int] = None):
    num_classes = len(dataset.classes)
    train_indices, val_indices = [], []
    for c in range(num_classes):
        idx = torch.where(torch.tensor(dataset.targets) == c)[0]
        split = int(len(idx) * train_size)
        train_indices.extend(idx[:split])
        val_indices.extend(idx[split:])
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def get_dataset_inverse_weighted_sampler(
    dataset: Dataset, dataset_targets: list, scale_dataset: int
) -> WeightedRandomSampler:
    """Returns a WeightedRandomSampler for the given dataset."""
    num_samples = int(len(dataset) * scale_dataset)
    # Balance the dataset defining weighted_sampler
    class_sample_counts = dict(Counter(dataset_targets))
    weights = 1.0 / torch.tensor(list(class_sample_counts.values()), dtype=torch.float)
    sample_weights = weights[dataset_targets]
    weighted_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=num_samples, replacement=True)
    return weighted_sampler


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", handlers=[logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)

    parser = ArgumentParser()
    parser.add_argument("-m", "--mode", type=str)
    parser.add_argument("-d", "--datasetpath", type=Path)
    parser.add_argument("-o", "--out", type=Path)
    parser.add_argument("--lr", default=4e-3, type=float)
    parser.add_argument("--only_classifier", action="store_true", default=False)
    parser.add_argument("--model_size", default="large")
    parser.add_argument("--batch", default=64, type=int)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--num_classes", default=0, type=int)
    parser.add_argument("--weight_decay", default=5e-3, type=float)
    parser.add_argument("--label_smoothing", default=0.1, type=float)
    parser.add_argument("--warmup", default=20, type=int)
    parser.add_argument("--scale_dataset", default=2, type=float)
    parser.add_argument("--temperature", default=3.0, type=float)
    parser.add_argument("--crop_size", default=0, type=int)
    parser.add_argument("--modelpath", type=Path)

    # Parse the arguments
    args = parser.parse_args()

    # Check if the path exists and is a directory
    if not args.datasetpath.exists():
        print(f"The path {args.path} does not exist.")
    elif not args.datasetpath.is_dir():
        print(f"The path {args.path} is not a directory.")
    # Check if the path exists and is a directory
    if not args.out.exists():
        args.out.mkdir(parents=True, exist_ok=True)

    if args.mode == "train":
        crop_size = None if args.crop_size <= 0 else args.crop_size
        transform = DEFAULT_CROP_VISDRONE_AUGMENT(crop_size)
        transform_val = DEFAULT_CROP_VISDRONE_AUGMENT_VAL(crop_size)
        # Modify the torchvision resnet model...
        dataset = ImageFolder(root=args.datasetpath, transform=transform)
        train_dataset, val_dataset = stratified_dataset_train_val_split(dataset, train_size=0.8, random_state=42)
        val_dataset.dataset.transform = transform_val
        train_targets = [
            dataset.targets[i] for i in tqdm(train_dataset.indices, desc="Inspecting train dataset targets")
        ]
        train_sampler = get_dataset_inverse_weighted_sampler(
            train_dataset, train_targets, scale_dataset=args.scale_dataset
        )

        args.num_classes = args.num_classes if args.num_classes > 0 else len(dataset.classes)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch,
            sampler=train_sampler,
            prefetch_factor=2,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            drop_last=True,
        )
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch, num_workers=2, drop_last=True)
        logger.info(f"Number of samples in training dataset: {len(train_dataset)}")
        logger.info(f"Number of samples in validation dataset: {len(val_dataset)}")

        # mean, std = _compute_mean_std_dataset(dataloader, method="subsample")

        # Define the model
        weights = "IMAGENET1K_V1" if args.only_classifier else None
        if args.model_size == "large":
            model = torchvision.models.convnext_large(weights=weights)
            model.classifier = nn.Sequential(
                LayerNorm2d((1536,), eps=1e-06, elementwise_affine=True),
                nn.Flatten(start_dim=1, end_dim=-1),
                nn.Linear(in_features=1536, out_features=1536 * 2, bias=True),
                nn.Linear(in_features=1536 * 2, out_features=args.num_classes, bias=True),
            )
        elif args.model_size == "base":
            model = torchvision.models.convnext_base(weights=weights)
            model.classifier = nn.Sequential(
                LayerNorm2d((1024,), eps=1e-06, elementwise_affine=True),
                nn.Flatten(start_dim=1, end_dim=-1),
                nn.Linear(in_features=1024, out_features=1024 * 2, bias=True),
                nn.Linear(in_features=1024 * 2, out_features=args.num_classes, bias=True),
            )
        elif args.model_size == "small":
            model = torchvision.models.convnext_small(weights=weights)
            model.classifier = nn.Sequential(
                LayerNorm2d((768,), eps=1e-06, elementwise_affine=True),
                nn.Flatten(start_dim=1, end_dim=-1),
                nn.Linear(in_features=768, out_features=768 * 2, bias=True),
                nn.Linear(in_features=768 * 2, out_features=args.num_classes, bias=True),
            )
        # set parameters to train
        param_size = 0
        if weights is None:
            for param in model.parameters():
                param.requires_grad = True
                param_size += param.numel()
        else:
            for param in model.parameters():
                param.requires_grad = False

            for param in model.classifier.parameters():
                param.requires_grad = True
                param_size += param.numel()

        print("-" * 80)
        intro = "START TRAINING"
        if NO_ASCII_ART:
            print(intro)
        else:
            ascii_art = pyfiglet.figlet_format(intro)  # ty: ignore
            print(ascii_art)
        print(
            f"Number of trainable parameters (milions): {param_size / 10**6:.2f}",
        )
        print("Training dataste size: ", len(dataset))
        # print("Computed mean and std of the dataset: ", mean.tolist(), std.tolist())
        print("-" * 80)
        print("=> Arguments: ")
        for i, arg_key in enumerate(vars(args)):
            print(f"{i:2}.{arg_key:15}= {vars(args)[arg_key]};")
        print("-" * 80)
        experiment_description = input("Additional Notes: ")
        args.experiment_description = experiment_description
        args.out = args.out / datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        args.out.mkdir(parents=True, exist_ok=True)
        # Save this run configuration
        with open(args.out / "config.json", "w+") as json_file:
            json.dump({str(v): str(vars(args)[v]) for v in vars(args)}, json_file, indent=4)

        model = train(
            model,
            train_dataloader,
            val_dataloader,
            lr=args.lr,
            epochs=args.epochs,
            label_smoothing=args.label_smoothing,
            warmup=args.warmup,
            weight_decay=args.weight_decay,
            temperature=args.temperature,
            checkpoint_folder_path=args.out,
        )
        torch.save(model, args.out / "end-training-model.pth")
    elif args.mode == "test":
        crop_size = None if args.crop_size <= 0 else args.crop_size
        transform = DEFAULT_CROP_VISDRONE_AUGMENT(crop_size)
        # Modify the torchvision resnet model...
        dataset = ImageFolder(root=args.datasetpath, transform=transform)
        num_samples = int(len(dataset.samples) * args.scale_dataset)
        # Balance the dataset defining weighted_sampler
        class_sample_counts = dict(Counter(dataset.targets))
        weights = 1.0 / torch.tensor(list(class_sample_counts.values()), dtype=torch.float)
        sample_weights = weights[dataset.targets]
        weighted_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=num_samples, replacement=True)
        num_classes = args.num_classes if args.num_classes > 0 else len(dataset.classes)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch,
            sampler=weighted_sampler,
            prefetch_factor=2,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )
        # mean, std = _compute_mean_std_dataset(dataloader, method="subsample")

        print("-" * 80)
        intro = "START TESTING"
        if NO_ASCII_ART:
            print(intro)
        else:
            ascii_art = pyfiglet.figlet_format(intro)  # ty: ignore
            print(ascii_art)
        print("Dataste size: ", len(dataset))
        metrics = {
            "accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=args.num_classes, top_k=1),
            "3-accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=args.num_classes, top_k=3),
            "5-accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=args.num_classes, top_k=5),
        }
        device = "cuda" if torch.cuda.is_available() else "cpu"
        test(args.modelpath, dataloader, device, args.out, metrics=metrics)
    else:
        print(f"No known mode '{args.mode}'")
