from pathlib import Path
import json
import torch

model_dir = Path("benchmark_assets/models/cifar10_resnet20")
model_dir.mkdir(parents=True, exist_ok=True)

model = torch.hub.load(
    "chenyaofo/pytorch-cifar-models",
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
#torch.save(model, model_dir / "model.pth")
torch.save(model.state_dict(), model_dir / "model_state_dict.pth")

info = {
    #"type": "torch_script",
    "type": "model_weights",
    #"type": "plain",
    "name": "cifar10_resnet20",
    "id": "cifar10_resnet20",
    "num_classes": 10,
    "task": "classification",
    "input_dimensionality": [3, 32, 32],
    "transformation": {
        "mean": [0.4914, 0.4822, 0.4465],
        "std": [0.247, 0.2435, 0.2616]
    }
}

with open(model_dir / "info.json", "w") as f:
    json.dump(info, f, indent=4)
