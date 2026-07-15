from pathlib import Path
from torchvision.datasets import CIFAR10

output_root = Path("benchmark_assets/datasets/cifar10_test")
output_root.mkdir(parents=True, exist_ok=True)

# select cifar 10 dataset in existing root
dataset = CIFAR10(
    root="/home/antonio-liguori/Documents/Projects/StableAI/nn_trust/data",
    train=False,
    download=False,
)

CLASS_NAMES = dataset.classes

class_counts = {class_name: 0 for class_name in CLASS_NAMES}

# prepare cifar10 dataset for ImageFolder format
for image, label in dataset:
    class_name = CLASS_NAMES[label]
    class_dir = output_root / class_name
    class_dir.mkdir(parents=True, exist_ok=True)

    image_id = class_counts[class_name]
    image.save(class_dir / f"{image_id:05d}.png")
    class_counts[class_name] += 1
