import json
from pathlib import Path
from typing import Union, Annotated, Literal, List, Optional
import logging

import torch
import torchvision
import yaml
from annotated_types import Gt, Ge
from pydantic import BaseModel, Field, field_validator, ValidationInfo
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import time
from nn_trust.attack import EvasionAttackFactory
from nn_trust.attack._evasion import EvasionAttack
from nn_trust.attack.evaluation._statistics import StatisticsFactory
from nn_trust.attack.evaluation.composer import ConfigStatisticComposer, StatisticComposer
from nn_trust.attack.utils._utils import enumerated_list
from nn_trust.attack.utils._utils import get_min
from nn_trust.attack.utils.logger import TensorboardLogger
from nn_trust.attack.utils.loss._loss import LossFactory
from nn_trust.attack.utils.loss.loss_composer import ConfigLossComposer, LossComposer
from nn_trust.core import Task, ModelAdapter
from .imagesaver import ImageSaver


class BenchmarkEvaluationConfig(BaseModel):
    statistics: list[str] = Field(default_factory=list)
    statistic_average_method: str


class BenchmarkOptionConfig(BaseModel):
    load_results: bool
    overwrite: bool
    num_images_to_save: int
    save_perturbation: bool
    gpu: bool
    output_path: str
    output_format: str

class BenchmarkDatasetTransformConfig(BaseModel):
    size: int
    crop: int
    transform_id: str
    mean: List[float]
    std: List[float]

class BenchmarkDatasetConfig(BaseModel):
    name: str
    num_classes: int
    subset: int
    batch: int
    type_dataset: int
    num_workers: int
    source_path: str
    transform_config: BenchmarkDatasetTransformConfig


class BenchmarkModelsConfig(BaseModel):
    name: str
    type: str
    pretrained: bool
    num_classes: int
    task: str


class BenchmarkAttackConfig(BaseModel):
    name: str
    max_iters: Optional[int] = None
    losses: Optional[List[str]] = None


class BenchmarkConfig(BaseModel):

    evaluation: BenchmarkEvaluationConfig
    options: BenchmarkOptionConfig
    datasets: List[BenchmarkDatasetConfig]
    models: List[BenchmarkModelsConfig]
    attacks: List[BenchmarkAttackConfig]

class EvaluatorConfig(BaseModel):
    """
    Configuration file for the Evaluator class
    """
    ################# GLOBAL #################
    model: ModelAdapter | str | torch.nn.Module= Field(default=...,
                                                        description='The model on which to generate the attack.')
    # list_models: list = Field(default=[],
    #                           description="List of models to test.")
    dataloader: DataLoader = Field(default=...,
                                   description="Dataset to use for the benchmarking.")

    attacks: List[BenchmarkAttackConfig] = Field(default_factory=list,
                                            description="List of attacks to perform. If None, all the attacks are performed.")
    losses: list = Field(default_factory=list,
                         description="List of losses to use in the test.")
    statistics: list = Field(default_factroy=list,
                             description="List of statistics names to use in the evaluation process.")
    ##########################################

    ################# OPTIONS #################
    load_results: bool = Field(default=False,
                               description="Load previous results and skip the tests that have already been done.")
    num_images_to_save: int = Field(default=10,
                                    description="Number of images to save.")
    save_perturbation: bool = Field(default=True,
                                    description="Whether to save the adversarial perturbation or not.")
    overwrite: bool = Field(default=True,
                            description="Whether to overwrite the results from the new tests onto the old one.")
    num_classes: Annotated[int, Ge(-1)] = Field(default=...,
                                               description="Number of possible classes")
    max_iters: Annotated[int, Gt(0)] = Field(default=100,
                                             description="Maximum number of iteration performed by each attack.")
    statistic_average_method: Literal["macro", "weighted", "micro"] = Field(default="micro",
                                                                            description="Torchmetrics aggregation method in case of multiclass task.")
    ###########################################

    ################# PATH #################
    output_path: str | Path = Field(default=Path("./benchmark_output"),
                     description="Path to the output folder.")
    output_format: Literal["report", "test"] = Field(default="report",
                                                     description="Output format: 'report' for saving results, 'test' for test mode.")
    logger: TensorboardLogger = Field(default=None,
                                      description="The logger to use for keeping track of the attacks.")
    ########################################

    ########################################
    inverse_transformation: torchvision.transforms.Compose = Field(default=None,
                                                                   description="Inverse transformation for showing the images.")
    device: torch.device = Field(default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                                 description='The device to use.')
    verbose: bool = Field(default=False,
                          description='True to generate more debug print.')

    class Config:
        arbitrary_types_allowed = True

    def __str__(self):
        return enumerated_list(self.__dict__, enumeration=False)

    @field_validator('device')
    @classmethod
    def set_device(cls, v: list) -> torch.device:
        """
        Set the device where the computation will be hold.
        """
        if isinstance(v, str):
            if v.lower() == 'cuda' or v.lower() == 'gpu':
                v = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                raise ValueError("The device must be 'cuda' or 'cpu'.")
        elif not isinstance(v, torch.device):
            raise TypeError("The device must be a string or a torch.device.")
        return v

    @field_validator('num_images_to_save', mode="after")
    @classmethod
    def valid_num_images(cls, v, info):
        # Need to access the current model values through info.data
        dataloader = info.data.get('dataloader')
        if dataloader is None:
            # Handle case where dataloader isn't available yet
            return v

        if v < 0:
            out = len(dataloader.dataset)
        else:
            out = min(v, len(dataloader.dataset))
        return out

    @field_validator('attacks')
    @classmethod
    def validate_attacks(cls, v, info:ValidationInfo):
        field_name = info.field_name
        complete_list = EvasionAttackFactory.list_attacks()
        return [atk for atk in v if atk.name in complete_list]

    @field_validator('losses', 'statistics')
    @classmethod
    def valid_list(cls, v, info: ValidationInfo):

        field_name = info.field_name
        if info.field_name == 'attacks':
            complete_list = EvasionAttackFactory.list_attacks()
        elif field_name == 'losses':
            complete_list = LossFactory.list_losses()
        else:
            complete_list = StatisticsFactory.list_statistics()

        if isinstance(v, list) and (len(v) == 0 or "none" in v):
            out = []
        elif v == "all" or "all" in v:
            # It is supported the two cases:
            # 1) statistics = ["all"]
            # 2) statistics = "all"

            out = complete_list
        else:
            # filter all the elements
            out = [item for item in complete_list if item in v]

            discarded_atk = [atk for atk in v if atk not in complete_list]
            if len(discarded_atk) > 0:
                print(
                    "The following {} are discarded because they don't exists: {}.".format(
                        info.field_name, discarded_atk)
                )
                if len(out) == 0:
                    raise ValueError("The list of attacks to perform is empty.")

        # print("The list of {} to use is {}".format(info.field_name, enumerated_list(out)))
        return out


class Evaluator:
    """
    This class computes the performance of a model against multiple attacks.
    """

    def __init__(self,
                 config: EvaluatorConfig,
                 **kwargs):
        """
        Initialize the evaluator with model, dataloader and attack configurations.

        Args:
            config: configuration class for setting all the necessary parameters.
            **kwargs: Additional parameters for attacks
        """
        self.config = config
        self.kwargs = kwargs
        self.image_saver = ImageSaver(
            name=config.model.name,
            output_path=config.output_path,
            num_images_to_save=config.num_images_to_save,
            plot_transformation=config.inverse_transformation,
            save_perturbation=config.save_perturbation,
            output_format=config.output_format,
        )

        # Initialize statistics composers
        cnf_composer = ConfigStatisticComposer(model=config.model.to(config.device),
                                               dataloader=config.dataloader,
                                               num_classes=config.num_classes,
                                               statistics=config.statistics,
                                               average_method=config.statistic_average_method
                                               )

        self.statistics_composer = StatisticComposer(config=cnf_composer)

        self.results = {
            'info': {},
            'metrics': {},
            'atk': {}
        }

    def evaluate_atk(self, 
                     task = None, 
                     dataset_name = None, 
                     model_name = None, 
                     attack_config = None, 
                     benchmark_progress = None,
                     atk: EvasionAttack = None) -> dict:
        """
        Evaluate the model on the attack that is passed.

            Args:
                atk: the attack that has to be performed.
        """

        # Process each batch
        for idx, _ in enumerate(tqdm(range(100))):
            current_progress = float((idx + 1) / 100)
            if task:
                task.update_state(
                    state='PROGRESS',
                    meta={
                        'progress': benchmark_progress,
                        'last_attack_performed':attack_config.name,
                        'is_over':False,
                        'dataset': dataset_name,
                        'model': model_name,
                        'attack_progress': current_progress
                    })
                if task.is_cancelled():
                    raise Exception("The task has been killed.")
                
            time.sleep(5)
#            batch = batch.to(self.config.device)
#            label = label.to(self.config.device)
#
#            # Generate adversarial examples
#            y_one_hot = torch.nn.functional.one_hot(label,
#                                                    num_classes=self.config.num_classes)
#            x_adv = atk.generate(
#                x=batch,
#                y=y_one_hot
#            ).detach()
#
#            ########### UPDATE STATISTICS ###########
#            with torch.no_grad():
#                out = self.config.model(batch)
#                out_adv = self.config.model(x_adv)
#
#            self.image_saver.save_images(img=batch,
#                                         img_adv=x_adv,
#                                         y=label,
#                                         y_pred=out,
#                                         y_pred_adv=out_adv,
#                                         element_info=element_info
#                                         )
#            input_stat = {
#                'x_adv': x_adv.detach(),
#                'x': batch.detach(),
#                'y': label,
#                'out': out,
#                'out_adv': out_adv,
#                'y_pred': out.argmax(-1),
#                'y_pred_adv': out_adv.argmax(-1),
#            }
#            self.statistics_composer.update(**input_stat)
#
#        return self.statistics_composer.compute()

    def evaluate(self, task,dataset_name, model_name):
        """
        Evaluate the model against all specified attacks and compute metrics.
        """

        # extract the general information
        batch, _, _ = next(iter(self.config.dataloader))
        self.results['info'] = {
            'name': self.config.model.name,
            'parameters': sum([param.numel() for param in self.config.model.parameters()]),
            'classes': self.config.num_classes,
            'dimensionality': batch[0].shape
        }
        self.config.model.to(self.config.device)
        self.config.model.eval()
        # TODO manage this expection iw we want less overhead and dont need statistics
        # TODO manage evaluation of base model within evaluator class, not in statistics compose
        if not self.config.output_format == "test":
            self.statistics_composer.update_global(model=self.config.model,
                                                dataloader=self.config.dataloader)

        ############################ TESTING VULNERABILITIES ############################
        for i,attack_config in enumerate(self.config.attacks):
            current_progress = float((i + 1) / len(self.config.attacks))
            if task:
                task.update_state(
                    state='PROGRESS',
                    meta={
                        'progress': current_progress,
                        'last_attack_performed':attack_config.name,
                        'is_over':False,
                        'dataset': dataset_name,
                        'model': model_name,
                        'attack_progress': 0.0
                    }
                )
            attack_config_dict = {k:v for k,v in attack_config.dict().items() if v is not None}
            atk_name = attack_config_dict.pop("name")
            time.sleep(20)
            self.evaluate_atk(task=task, 
                              dataset_name=dataset_name, 
                              model_name=model_name, 
                              attack_config=attack_config,
                              benchmark_progress=current_progress)
            #try:
            #    self.image_saver.new_attack(atk_name=atk_name)
            #    self.atk_name = atk_name
#
            #    if "losses" in attack_config_dict:
            #        # If losses are specified, convert them to Loss objects
            #        attack_config_dict['loss'] = LossComposer(ConfigLossComposer(
            #            loss=attack_config_dict['losses'],
            #            p = attack_config_dict.get('p', 2.0),
            #            loss_weights=attack_config_dict.get('loss_weights', [1.0] * len(attack_config_dict['losses'])),
            #        ))
#
            #    atk = EvasionAttackFactory.create_attack(atk_name,
            #                                            model=self.config.model,
            #                                            device=self.config.device,
            #                                            task=Task.Classification,
            #                                            targeted=False,
            #                                            **attack_config_dict
            #                                            )
            #    if self.config.model.task not in atk.TASKS:
            #        logging.warning(f"\U0001F928 Attack {atk_name} does not support Model {self.config.model.name} task {self.config.model.task}. Skipped Execution.")
            #        continue
#
            #    self.results['atk'][atk_name] = self.evaluate_atk(atk=atk)
            #except Exception as e:
            #    print(f"\U0001F975 Execution of attack '{atk_name}' on model '{self.config.model.name}' has failed: '{e}'")
            #torch.cuda.empty_cache()

        #print("################### FINISH the attacks ####################")

        #self.results['metrics'] = self.statistics_composer.compute_global_state()

    def save_results(self):
        """
        Save evaluation results to a JSON file
        """
        # Updating the list of attack with the one that have already been done
        result_file = Path(self.config.output_path) / self.config.model.name / 'data.json'
        # out = self.results
        if self.config.load_results and result_file.exists():
            with open(result_file, 'r') as f:
                # The YAML file format is unusual with [evaluation] section header
                # We need to handle this custom format
                results = yaml.safe_load(f)

            # Merge the previous results with the new one
            for section in ['metrics', 'atk']:
                for key, value in results[section].items():
                    if (key in self.results[section]) and (key != "confusion_matrix"):
                        # If key exists in both dictionaries, take the minimum value
                        self.results[section][key] = get_min(self.results[section][key], value)
                    else:
                        # If key only exists in dict2, add it to the out
                        self.results[section][key] = value

        result_file.parent.mkdir(exist_ok=True, parents=True)
        with open(result_file, 'w') as f:
            json.dump(self.results, f)
