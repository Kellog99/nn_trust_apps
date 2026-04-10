import pathlib

import torch
from transformers import AutoModelForImageClassification

from nn_trust.core import ModelAdapter, Knowledge, Task


class HFModel(ModelAdapter):

    def __init__(self,
                 model_name: str,
                 checkpoint_path: str | pathlib.Path | None = None,
                 task: str | Task = "classification",
                 knowledge: str | Knowledge = "white",
                 num_labels: int = None,
                 **kwargs
                 ):
        super().__init__(model=None)

        if checkpoint_path is not None:
            self.model = AutoModelForImageClassification.from_pretrained(model_name, num_labels=num_labels,
                                                                         ignore_mismatched_sizes=True)
            weight_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))["state_dict"]
            if list(weight_dict.keys())[0].startswith("model.timm_model."):
                weight_dict = {k.replace("model.timm_model.", ""): v for k, v in weight_dict.items()}
            weight_dict = {k.replace("model.timm_model.", ""): v for k, v in weight_dict.items()}
            res = self.model.timm_model.load_state_dict(weight_dict, strict=False)
            assert len(res.missing_keys) == 0, f"Missing keys when loading checkpoint: {res.missing_keys}"
        else:
            self.model = AutoModelForImageClassification.from_pretrained(model_name)
        self.name = model_name
        self.task = Task.from_str(task) if isinstance(task, str) else task
        self.knowledge = Knowledge.from_str(knowledge) if isinstance(knowledge, str) else knowledge

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.task == Task.Classification:
            return self.model(x).logits
        else:
            return NotImplementedError

    #def to(self, device: torch.device):
    #    self.model = self.model.to(device)
    #    return self