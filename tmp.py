import datetime
import json
from pathlib import Path

from models.main_model import ServerConfig

config = ServerConfig()

timmModels = [
    {
        "name": 'resnet50',
        "task": 'classification',
        "domain": 'computer_vision',
        "classes": 1000,
        "parameters": 25557032,
        "input_dimensionality": [3, 224, 224],
        "dataset": 'ImageNet-1k',
        "description": 'ResNet-50 deep residual network'
    },
    {
        "name": 'efficientnet_b0',
        "task": 'classification',
        "domain": 'computer_vision',
        "classes": 1000,
        "parameters": 5288548,
        "input_dimensionality": [3, 224, 224],
        "dataset": 'ImageNet-1k',
        "description": 'EfficientNet-B0 compound scaling network'
    },
    {
        "name": 'vit_base_patch16_224',
        "task": 'classification',
        "domain": 'computer_vision',
        "classes": 1000,
        "parameters": 86567656,
        "input_dimensionality": [3, 224, 224],
        "dataset": 'ImageNet-21k',
        "description": 'Vision Transformer Base with 16x16 patches'
    },
    {
        "name": 'convnext_tiny',
        "task": 'classification',
        "domain": 'computer_vision',
        "classes": 1000,
        "parameters": 28589128,
        "input_dimensionality": [3, 224, 224],
        "dataset": 'ImageNet-1k',
        "description": 'ConvNeXt Tiny modernized ConvNet'
    },
    {
        "name": 'swin_tiny_patch4_window7_224',
        "task": 'classification',
        "domain": 'computer_vision',
        "classes": 1000,
        "parameters": 28288354,
        "input_dimensionality": [3, 224, 224],
        "dataset": 'ImageNet-1k',
        "description": 'Swin Transformer Tiny with hierarchical architecture'
    },
    {
        "name": 'mobilenetv3_large_100',
        "task": 'classification',
        "domain": 'computer_vision',
        "classes": 1000,
        "parameters": 5483032,
        "input_dimensionality": [3, 224, 224],
        "dataset": 'ImageNet-1k',
        "description": 'MobileNetV3 Large efficient mobile network'
    },
    {
        "name": 'densenet121',
        "task": 'classification',
        "domain": 'computer_vision',
        "classes": 1000,
        "parameters": 7978856,
        "input_dimensionality": [3, 224, 224],
        "dataset": 'ImageNet-1k',
        "description": 'DenseNet-121 densely connected network'
    },
    {
        "name": 'regnetx_002',
        "task": 'classification',
        "domain": 'computer_vision',
        "classes": 1000,
        "parameters": 2684792,
        "input_dimensionality": [3, 224, 224],
        "dataset": 'ImageNet-1k',
        "description": 'RegNetX-002 designed network space'
    }
]

repo = Path(config.path_model_repo)
for model in timmModels:
    model["id"] = model["name"]
    model_path = repo / model["name"]
    date = datetime.datetime.now()
    model["date"] = date.strftime("%Y-%m-%d")
    model["api"] = "timm"
    if not model_path.exists():
        model_path.mkdir(parents=True)
    with open(model_path / "info.json", "w") as file:
        json.dump(model, file)
