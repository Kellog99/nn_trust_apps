from pathlib import Path
from typing import Optional

import gradio as gr
import torch
import torchvision.transforms.functional as F

from .imageloader import from_standardized_image_to_image, rescale_image_range


class OutputImage:
    def __init__(self,
                 log_path: Path,
                 labels2id=dict[int, str],
                 target_class: torch.Tensor = None
                 ):
        super().__init__()
        self.log_path = log_path
        self.labels2id = labels2id
        self.id2labels = {label: class_id for class_id, label in labels2id.items()}
        self.target_class = target_class

    def generate(self):
        with gr.Row(equal_height=True):
            with gr.Column():
                self.img_display1 = gr.Image(
                    label="ORIGINAL IMAGE",
                    format="png",
                    interactive=False,
                    width=400,
                    height=400)
                self.textbox1 = gr.Textbox(
                    interactive=False,
                    lines=1,
                    container=False,
                )
            self.img_display2 = gr.Image(label="PERTURBATION",
                                         format="png",
                                         interactive=False,
                                         width=400,
                                         height=400)
            with gr.Column():
                self.img_display3 = gr.Image(label="ADVERSARIAL IMAGE",
                                             format="png",
                                             interactive=False,
                                             width=400,
                                             height=400)
                self.textbox3 = gr.Textbox(
                    interactive=False,
                    lines=1,
                    container=False,
                )
        return self.img_display1, self.textbox1, self.img_display2, self.img_display3, self.textbox3

    def update(self, step_idx: int) -> list:
        try:
            data = torch.load(self.log_path, weights_only=True)
            max_step = len(data.get("generate/res", [])) - 1
            step_idx = max(min(step_idx, max_step), 0)
        except Exception as e:
            raise gr.Error("No data to load") from e

        # Perturbation
        perturbation = data["generate/perturbation"][step_idx][0]
        perturbation = rescale_image_range(perturbation)
        perturbation = F.to_pil_image(perturbation)

        # Adversarial image
        adversarial_img = data["generate/res"][step_idx][0]
        adversarial_img = from_standardized_image_to_image(adversarial_img)
        adversarial_img = F.to_pil_image(adversarial_img)

        # Compare original and adversarial classification
        adversarial_prediction = data["generate/model_adv_classification"][step_idx]
        og_prediction = data["generate/original_classification"][0]

        # Format the adversarial prediction label
        """import pdb
        pdb.set_trace()"""
        target_class = torch.tensor([self.id2labels[self.target_class]]).cpu() if self.target_class else None
        if target_class is not None:
            elem_classes = ["cls-label", "atk-fail"] if torch.all(torch.eq(adversarial_prediction, target_class)) else [
                "cls-label", "atk-success"]
        else:
            elem_classes = ["cls-label", "atk-fail"] if torch.all(
                torch.ne(adversarial_prediction, og_prediction)) else ["cls-label", "atk-success"]
        adversarial_prediction = self.labels2id.get(adversarial_prediction.item(), "Unknown class").upper()
        return [
            gr.update(value=perturbation),
            gr.update(value=adversarial_img),
            gr.update(value=adversarial_prediction, elem_classes=elem_classes),
        ]
