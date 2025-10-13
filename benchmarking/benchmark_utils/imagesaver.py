import shutil
from pathlib import Path
from typing import Annotated, Union, List

import torch
from annotated_types import Ge
from torchvision.transforms import Compose, ToPILImage


class ImageSaver:
    """
    This class has the goal to organize the saving of the images.
    The global organization of a folder is the following:
        * output folder/
        * atks_info.json 			# It contains the information of all the attacks.
        * name dataset/
            * model1/
                - examples/
                    * atk1/
                        - id1_ypred_yadvpred.png
                        - id1_pert.png
                        - ...
                    * atk2/
                        - id1_ypred_yadvpred.png
                        - id1_pert.png
                        - ...
                - data.json
            * model2/
                - attacks/
                    * atk1/
                        - adv1.png
                        - perturbation1.png
                        - ...
                    * atk2/
                        - adv1.png
                        - perturbation1.png
                        - ...
                - data.json
            * ...
            * examples/
                * id1_y1.png
                * id2_y2.png
                * ...
            * name_classes.json
    """

    def __init__(self,
                 name: str,
                 num_images_to_save: Annotated[int, Ge(0)],
                 plot_transformation: Union[Compose | ToPILImage],
                 output_path: Path | str,
                 output_format: str = "report",
                 save_perturbation: bool = True):

        self.num_images = num_images_to_save if num_images_to_save >= 0 else float('inf')

        if isinstance(output_path, str):
            output_path = Path(output_path)

        self.attacks = output_path / f'{name}/examples'  # folder where the adversarial images are stored
        self.examples = output_path / 'examples'  # folder where the true images are stored
        self.save_perturbation = save_perturbation

        self.plot_transformation = plot_transformation
        self.atk_name = None
        self.output_format = output_format

        # Deletion of all the previous images
        if self.attacks.exists():
            shutil.rmtree(self.attacks)
        if self.examples.exists():
            shutil.rmtree(self.examples)

        self.attacks.mkdir(parents=True, exist_ok=True)
        self.examples.mkdir(parents=True, exist_ok=True)

        # set the counter of the saved images to 0.
        self.img = 0
        self.adv_images = 0

    def _transform(self, img: torch.Tensor):
        """
        Transform the image according to the transformation that was given in the init
        """
        return img if self.plot_transformation is None else self.plot_transformation(img)

    def new_attack(self, atk_name: str):
        """
        Since it gets generated just once, the counter for the true images is 0 only when the class is initialized.
        After that only the counter of the adversarial images gets reset.
        """
        if self.num_images == float('inf'):
            self.num_images = self.adv_images

        self.adv_images = 0
        self.atk_name = atk_name.removesuffix("Attack")

        # Create the folder for the adversarial images for the attack `atk_name`
        (self.attacks / self.atk_name).mkdir(parents=True, exist_ok=True)

    def save_images(self,
                    img: torch.Tensor,
                    img_adv: torch.Tensor,
                    y: torch.Tensor,
                    y_pred: torch.Tensor,
                    y_pred_adv: torch.Tensor,
                    element_info: List[dict] = None
                    ) -> None:
        """
        It saves the first `num_images` adversarial images
        """
        if img.shape != img_adv.shape:
            raise ValueError("The set of original images and the modified one, have different shape.")
        if y_pred.shape != y_pred_adv.shape:
            raise ValueError("The number of predictions and the number of adversarial predictions is different.")

        # The first dimension tells the number of element of the batch
        pred_labels = y_pred.argmax(-1).tolist()
        pred_adv_labels = y_pred_adv.argmax(-1).tolist()

        if self.output_format == "report":
            for i in range(img.shape[0]):
                if self.img < self.num_images:
                    # Saving reference images.
                    self._transform(img[i]).save(
                        self.examples / "{}_{}.png".format(self.img,
                                                        pred_labels[i])
                    )
                    self.img += 1

                if self.adv_images < self.num_images:
                    # Saving adversarial images.
                    self._transform(img_adv[i]).save(
                        self.attacks / "{}/{}_{}_{}.png".format(self.atk_name,
                                                                self.adv_images,
                                                                pred_labels[i],
                                                                pred_adv_labels[i])
                    )
                    if self.save_perturbation:
                        self._transform(img_adv[i] - img[i]).save(
                            self.attacks / "{}/{}_pert.png".format(self.atk_name,
                                                                self.adv_images)
                        )
                    self.adv_images += 1
        elif self.output_format == "test":
            for i in range(img.shape[0]):
                # succesful_attack = (y[i] != pred_labels[i]) and (y[i] != pred_adv_labels[i])

                suffix = f"{y[i]}_{pred_labels[i]}_{pred_adv_labels[i]}"
                suffixed_path = f"{Path(element_info['path'][i])}_{suffix}.png"
                full_path = self.attacks / self.atk_name / suffixed_path
                full_path.parent.mkdir(parents=True, exist_ok=True)
                self._transform(img_adv[i]).save(full_path)
        else:
            raise ValueError(f"Unknown output format: {self.output_format}")
