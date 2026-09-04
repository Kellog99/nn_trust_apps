from pathlib import Path
import json
import torch

model_dir = Path("benchmark_assets/model/cifar10_resnet20")
model_dir.mkdir(parents=True, exist_ok=True)

model = torch.hub.load(
    "chenyaofo/pytorch-cifar-model",
    "cifar10_resnet20",
    pretrained=True,
    trust_repo=True,
    skip_validation=True,
)

model.eval()

# Run the model's forward method with the provided example_inputs. As each operation executes, PyTorch logs it. The resulting ScriptModule essentially contains a frozen snapshot of the computation graph generated during that single forward pass.
example_input = torch.randn(1, 3, 32, 32)

with torch.no_grad():
    traced_model = torch.jit.trace(model, example_input)

#traced_model.save(model_dir / "model.pth")
torch.save(model, model_dir / "model.pth")

info = {
    #"type": "torch_script",
    "type": "plain",
    "name": "cifar10_resnet20",
    "num_classes": 10,
    "task": "classification",
    "input_size": 32
}

with open(model_dir / "info.json", "w") as f:
    json.dump(info, f, indent=4)
