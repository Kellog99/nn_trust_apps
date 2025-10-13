"""
TITANN: Tool for Inspection and Trustworthiness Assessment of Neural Networks

A Gradio-based interface for testing adversarial attacks on image classification models.
"""

import json
import os
import logging
import time
from datetime import datetime
from pathlib import Path
import warnings

from argparse import ArgumentParser

import gradio as gr
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet101, ResNet101_Weights

from blocks.attacks import ListAttacks
from blocks.imageloader import InputImage, from_standardized_image_to_resnet
from blocks.metrics import MetricsCalculator
from blocks.outputimage import OutputImage
from nn_trust.attack.utils.logger import PyTorchCheckpointLogger
from nn_trust.core import ModelAdapter, Task


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class TitanApp:
    """
    This class has the role to organize all the macro components that will be used in the Demo.
    """

    def __init__(
        self,
        device: torch.device,
        model: ModelAdapter,
        task: int = Task.Classification,
        css_path: Path = Path("./style.css"),
        log_path: Path = Path("./titann-app-logs/log"),
        logo_path: Path = Path("./titann-logo.png"),
        classes_path: Path = Path("./classes.json"),
        server_name: str = "0.0.0.0",
        server_port: int = 7860,
    ):
        """
        Args:
            device: Device where the algorithms will run.
            model: Model that will be tested.
            css_path: Path to the css file that define the style of the App.
            log_path: Path to the folder where the results will be saved.
            logo_path: Path to the logo
            classes_path: Path to the name of the classes
            server_name:
            server_port:
        """
        self.device = device
        self.model = model.to(self.device)
        self.log_path = log_path
        self.out_path = log_path / f"{datetime.now().strftime('%Y%m%dT%H%M%S')}.pth"
        self.logo_path = logo_path
        self.server_name = server_name
        self.server_port = server_port
        self.num_step = 0

        ######################## upload the style of the whole UI ########################
        if isinstance(css_path, Path):
            css_path = css_path
        elif isinstance(css_path, str):
            css_path = Path(css_path)
        else:
            raise ValueError(f"The type {type(css_path)} is not supported.")

        if css_path.exists():
            with open(css_path, "r") as f:
                self.css = f.read()
        else:
            raise ValueError("The path to the Style file is not valid. ")
        #########################################################################

        # Load labels id
        if classes_path.is_file():
            with open(classes_path) as f:
                data = json.load(f)
        else:
            raise ValueError(f"The path '{classes_path}' does not exists.")
        self.labels_id = {_id: label_name.split(",")[0] for _id, (key, label_name) in enumerate(data.items())}
        self.num_classes = len(self.labels_id)
        logging.info(f"Loaded class label file with {self.num_classes} classes")

        ############ UI macro components ############
        # Input image to attack
        self.input_image_block = InputImage(model=self.model, device=self.device, labels_id=self.labels_id)
        # List of attack to perform
        self.list_attacks = ListAttacks(model=self.model, device=self.device, task=model.task, labels_id=self.labels_id)
        # Output Images
        self.output = OutputImage(log_path=self.out_path, labels2id=self.labels_id)
        # List of metrics to display
        self.metrics = MetricsCalculator()
        self.attack_time = 0.0

        logging.info(f"TITANN App succesfully initialized.")

    @staticmethod
    def disable_feature(input):
        # Enable button only if image is not None
        return gr.update(interactive=input is not None)

    def execute_attack(self,
                       target: str,
                       targeted: bool,
                       progress=gr.Progress(track_tqdm=True)):
        """
        This function takes the configurator with the parameters as specified by the gradio interface,
        the image and then proceed to attack, then using the PyTorchCheckpointLogger
        we have a loadable file each 30 s or 10 iterations.
        """
        logging.info(f"Starting attack execution: [{self.list_attacks.atk}].")
        # Taking the image and its original predicted label that has to be tested
        x = self.input_image_block.get_image(format="standardized")
        y = self.input_image_block.get_prediction(format="id")

        # Taking the attack that has to be performed
        attack = self.list_attacks.get_attack()

        # Create the adversarial sample
        if self.out_path.is_file() and self.out_path.stat().st_size > 0:
            self.out_path.unlink()

        logger = PyTorchCheckpointLogger(path=Path(self.out_path), states=["generate"], max_size=100)
        if attack.config.targeted:
            y = self.list_attacks.target_class

        #################### ATTACK ####################
        start_time = time.time()
        _ = attack.generate(x=x,
                            y=F.one_hot(y, num_classes=self.num_classes).float(),
                            logger=logger,
                            progress=progress)
        self.metrics.time = time.time() - start_time
        ################################################

        logger.close()
        logging.info("Completed attack execution.")

        if self.out_path.exists():
            checkpoint = torch.load(self.out_path)
        else:
            raise ValueError(f"There is no checkpoint in {self.out_path}. File does not exist.")

        self.num_step = len(checkpoint["generate/confidence"])

        setattr(self.metrics, "confidence", [el.item() for el in checkpoint["generate/confidence"]])
        setattr(self.metrics, "most_confidence", [el.item() for el in checkpoint["generate/most_confident"]])
        setattr(self.metrics, "target", target if targeted else None)
        setattr(self.output, "target_class", target if targeted else None)

        # return gr.update(value = "✅ Attack finish")
        return gr.update(visible=False)

    def first_output(self):
        """
        After having done an attack, this function will populate all the elements that are needed in the output part.
        """

        # Adversarial image
        x = self.input_image_block.get_image(format="pil")
        y = self.input_image_block.get_prediction(format="label")
        out = [gr.update(value=x), gr.update(value=y, elem_classes=["cls-label", "atk-success"])]

        out = out + self.output.update(step_idx=self.num_step - 1)
        out.append(gr.Markdown(f"""
                                    This plot shows the changes of two confidences over the iterations:\n
                                    1) The blue represents the {self.metrics.target if self.metrics.target is not None else 'most probable'} confidence.\n 
                                    2) The green plot represent the confidence of the original class.\n 
                                    It can be seen that with each iteration the confidence of the original class decreases, showing how each iteration improves the effectiveness of the attack.
                                    """)
                   )
        out.append(self.metrics.update(step_idx=self.num_step - 1))

        out.append(
            gr.update(minimum=0,
                      maximum=self.num_step,
                      interactive=True,
                      value=self.num_step)
        )
        return out

    def update(self, step_idx: int):
        out = self.output.update(step_idx=step_idx - 1)
        out.append(self.metrics.update(step_idx=step_idx - 1))
        return out

    def generate(self):
        with gr.Blocks(
                head=f'<link rel="icon" href="{self.logo_path}" type="image/x-icon">',
                css=self.css,
                title="TITANN"
        ) as demo:
            gr.HTML("""
                <h1><b>TITANN: Tool for Inspection and Trustworthiness Assessment of Neural Networks</b></h1>
                </center>
                <br>
                """)

            with gr.Row(scale=1):
                with gr.Column(scale=3, elem_classes="column-input"):
                    gr.Markdown("""
                    # RUN EVASION ATTACK
                    Step to do for executing an attack:\n
                    1. Select Image\n  
                    2. Choose the Attack\n
                    3. Configure Attack's parameters\n
                    4. Fire!
                    """)
                    input_image = self.input_image_block.generate()
                    targeted, target = self.list_attacks.generate()
                    btn_fire = gr.Button("🔥 Fire", interactive=False)
                    progress_bar = gr.Textbox(value="", label="", info="", visible=False, interactive=False)

                with gr.Column(scale=7, visible=False) as out:
                    gr.Markdown("# ATTACK RESULTS")
                    img1, txt1, img2, img3, txt3 = self.output.generate()
                    with gr.Accordion("More results", open=False):
                        gr.Markdown("## Attack statistics")
                        desc, confidence_plot, metrics = self.metrics.generate()
                        slider = gr.Slider(interactive=False)

                    img1.change(fn=self.disable_feature,
                                inputs=img1,
                                outputs=slider)

                    slider.change(
                        fn=self.update,
                        inputs=[slider],
                        outputs=[img2, img3, txt3, confidence_plot]
                    )

                    img3.change(self.metrics.update_metrics,
                                inputs=[img1, img3],
                                outputs=metrics)

            ####################  Attack Execution ####################
            # To do the attack and to visualise the progress bar, it is necessary to do some steps:
            # 1) Make the textbox, where the progress bar will be displayed, visible
            # 2) Execute the attack
            # 3) Make not visible the textbox
            # 4) Populate the output block

            btn_fire.click(
                fn=lambda: gr.update(visible=True),
                outputs=progress_bar
            ).success(
                fn=self.execute_attack,
                inputs=[target, targeted],
                outputs=progress_bar
            ).success(
                fn=lambda: gr.update(visible=True),
                outputs=out
            ).success(
                fn=self.first_output,
                outputs=[img1, txt1, img2, img3, txt3, desc, confidence_plot, slider],
            )
            ###########################################################

            input_image.change(fn=self.disable_feature,
                               inputs=input_image,
                               outputs=btn_fire)

            # (The rest of the Pydantic models, generate, process_and_display_outputs, tensor_to_pil, etc. remain the same)
            logging.info(f"TITANN APP is accessible at http://{self.server_name}:{self.server_port}/?__theme=dark")
            # Remove the previous log if exists and exit the app
            self.out_path.unlink(missing_ok=True)
            return demo


if __name__ == "__main__":
    warnings.simplefilter(action='ignore', category=FutureWarning)
    # Initialize Logging
    handler = logging.StreamHandler()
    handler.addFilter(lambda record: record.name == "root")
    logging.basicConfig(level=logging.INFO, handlers=[handler])
    plt.switch_backend('agg')

    ############## UI creation ##############

    #########################################

    # Parse args in case of additional configuration might be needed
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=Path, default=Path("./model.pth"))
    parser.add_argument("--labels_path", type=Path, default=Path("./classes.json"))
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
    else:
        device = args.device

    # Initialize model and device
    available_models = os.listdir("./assets/models")
    for i, model in enumerate(available_models):
        print(f"{i}: {model}")
    selected_model_id = input("Select a model")
    model_path = Path("./assets/models") / available_models[int(selected_model_id)]
    device = torch.device(device)
    logging.info(f"Device selected: {device}")
    model = torch.load(str(model_path), weights_only=False).eval()
    model = ModelAdapter(model=model, task=Task.Classification, transform=from_standardized_image_to_resnet)

    # Start the app
    app = TitanApp(classes_path=args.labels_path, device=device, model=model)
    demo = app.generate()
    demo.launch(
        debug=False,
        favicon_path=app.logo_path,
        server_name=app.server_name,
        server_port=app.server_port,
        quiet=True,
    )
    demo.close()
    app.generate()
