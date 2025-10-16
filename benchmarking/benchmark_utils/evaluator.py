import json
import pathlib
from pathlib import Path
from typing import Union, Annotated, Literal, List, Optional
import logging
import time
from tqdm.auto import tqdm
import yaml
import glob
import re
import os
import traceback
import pickle

import torch
import torchvision
from annotated_types import Gt, Ge
from pydantic import BaseModel, Field, field_validator, ValidationInfo
from torch.utils.data import DataLoader
from inspect import signature
from nn_trust.attack import EvasionAttackFactory
from nn_trust.attack._evasion import EvasionAttack
from nn_trust.attack.evaluation._statistics import StatisticsFactory
from nn_trust.attack.evaluation.composer import ConfigStatisticComposer, StatisticComposer
from nn_trust.attack.utils._utils import enumerated_list, get_min
from nn_trust.attack.utils.logger import TensorboardLogger
from nn_trust.attack.utils.loss._loss import LossFactory
from nn_trust.attack.utils.loss.loss_composer import ConfigLossComposer, LossComposer
from nn_trust.core import Task, ModelAdapter
from .config import get_data_transformation_config
from .utils import get_dataloader, get_model
from typing import Dict


class Plan:

    def __init__(self, 
                 worker_action : callable, 
                 worker_params : Dict[str,Dict], 
                 action : callable = None,
                 params : Dict = None):
        """
        This class stores a callable to be executed with a list of parameters.
        """
        self.action = action
        self.params = params
        self.worker_action = worker_action
        self.worker_params = worker_params

    def __repr__(self):
        return f'Plan(worker_action={self.worker_action.__name__}, num_attacks={len(self.worker_params)}, params={self.params}, worker_params={self.worker_params})'

class BenchmarkEvaluationConfig(BaseModel):
    statistics: list[dict] | None = Field(default_factory=lambda x: [])

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
    id: str | None = None
    max_iters: int | None = None
    losses: List[str] | None = None


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
    model: ModelAdapter | str | torch.nn.Module = Field(default=...,
                                                        description='The model on which to generate the attack.')
    dataloader: DataLoader | str = Field(default=...,
                            description="Dataset to use for the benchmarking.")
    attacks: List[BenchmarkAttackConfig] = Field(default_factory=list,
                            description="List of attacks to perform. If None, all the attacks are performed.")
    statistics: list[dict] = Field(default_factroy=list,
                            description="List of statistics names to use in the evaluation process.")
    load_results: bool = Field(default=False,
                            description="Load previous results and skip the tests that have already been done.")
    save_perturbation: bool = Field(default=True,
                            description="Whether to save the adversarial perturbation or not.")
    overwrite: bool = Field(default=True,
                            description="Whether to overwrite the results from the new tests onto the old one.")
    num_classes: Annotated[int, Ge(-1)] = Field(default=...,
                            description="Number of possible classes")
    output_path: str | Path = Field(default=Path("./benchmark_output"),
                            description="Path to the output folder.")
    output_format: Literal["report", "test"] = Field(default="report",
                            description="Output format: 'report' for saving results, 'test' for test mode.")
    logger: TensorboardLogger = Field(default=None,
                            description="The logger to use for keeping track of the attacks.")
    inverse_transformation: torchvision.transforms.Compose = Field(default=None,
                            description="Inverse transformation for showing the images.")
    device: torch.device = Field(default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                            description='The device to use.')
    verbose: bool = Field(default=False,
                            description='True to generate more debug print.')
    reference: str = Field(default='identitybaseline', description='Include an attack to use as reference. Usually its the identity')

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

    @field_validator('attacks')
    @classmethod
    def validate_attacks(cls, v, info:ValidationInfo):
        field_name = info.field_name
        complete_list = EvasionAttackFactory.list_attacks()
        return [atk for atk in v if atk.name in complete_list]


class Evaluator:
    """
    This class computes the performance of a model against multiple attacks.
    """

    def __init__(self,
                 config: EvaluatorConfig | None = None,
                 **kwargs):
        """
        Initialize the evaluator with model, dataloader and attack configurations.

        self.results

        Args:
            config: configuration class for setting all the necessary parameters.
            **kwargs: Additional parameters for attacks
        """
        self.config = config
        self.kwargs = kwargs

        self.results = {
            'info': {},  # general information about dataset and model
            'attacks': {},  # results of attacks evaluation
            'aggregate_statistics': {}  # Aggregate information from evaluated attacks if available (optional)
        }

        self.config.model.to(self.config.device)
        self.config.model.eval()

        if not "reference" in [x.id for x in self.config.attacks]:
            self.config.attacks.append(BenchmarkAttackConfig(
                name=self.config.reference,
                id='reference'
            ))

    @classmethod
    def from_config(cls, config: dict, dataset: dict, model_config: dict):
        """Standard instantiation method tor Evaluator class.
        Take as input a config file, and specify a model and a dataset to evaluate within the given configuration.
        Return an evaluator that can runt adversarial test attacks and produce benchmark outputs
        """
        transform, inverse_transform = get_data_transformation_config(
            transform_id=dataset["transform_config"]["transform_id"],
            size=dataset["transform_config"]["size"],
            crop=dataset["transform_config"].get("crop"),
            mean=dataset["transform_config"].get("mean"),
            std=dataset["transform_config"].get("std"),
        )

        dataloader = get_dataloader(
            dataset=dataset["source_path"],
            batch=dataset["batch"],
            subset=dataset["subset"],
            type_dataset=dataset["type_dataset"],
            transform=transform,
            num_workers=dataset["num_workers"],
            name=dataset["name"]
        )

        model = get_model(
            model_name=model_config.get("name"),
            model_type=model_config.get("type"),
            model_weights_path=model_config.get("weights_path", None),
            mean=dataset["transform_config"].get("mean"),
            std=dataset["transform_config"].get("std"),
            model_task=model_config.get("task")
        )

        num_classes = (
            model_config["num_classes"]
            if model_config["num_classes"] > 0
            else len(dataloader.dataset.dataset.classes)  # TODO: Manage both datasets and subsets
        )

        print(f"This is options configurations {config['options']}")

        return cls(
            config=EvaluatorConfig(
                model=model,
                dataloader=dataloader,
                statistics=config["evaluation"]["statistics"],
                inverse_transformation=inverse_transform,
                attacks=config["attacks"],
                save_perturbation=config["options"]["save_perturbation"],
                overwrite=config["options"]["overwrite"],
                num_classes=num_classes,
                output_path=config["output_path"],
                output_format=config["options"]["output_format"],
            )
        )

    @staticmethod
    def evaluate_attack(
            dataloader: torch.utils.data.DataLoader,
            model: ModelAdapter,
            attack_config: dict,
            statistics: list[dict],
            device: torch.device,
            num_classes: int,
            verbose: bool = False,
        ) -> dict:
        """
        Evaluate the model on the attack that is passed.

            Args:
                atk: the attack that has to be performed.
        """
        # INIT MODEL , DATA, STATISTICCOMPOSER, ATTACK
        ## 1. STATISTICCOMPOSER
        statistics_composer = StatisticComposer(config=ConfigStatisticComposer(
            statistics=statistics,
            num_classes=num_classes
        ))
        ## 2. ATTACK
        atk_name = attack_config.pop("name")
        atk_id = attack_config.pop("id", atk_name)
        if "losses" in attack_config:
            # If losses are specified, convert them to Loss objects
            attack_config['loss'] = LossComposer(ConfigLossComposer(
                loss=attack_config['losses'],
                p=attack_config.get('p', 2.0),
                loss_weights=attack_config.get('loss_weights', [1.0] * len(attack_config['losses'])),
            ))
        atk = EvasionAttackFactory.create_attack(
            atk_name,
            model=model,
            device=device,
            task=Task.Classification,
            targeted=False,
            **attack_config
        )
        atk.name = atk_id
        if model.task not in atk.TASKS:
            raise ValueError(
                f"\U0001F928 Attack {atk_name} does not support Model {model.name} task {model.task}.")

        ### PREPARE EXECUTION
        if verbose:
            progress_bar = enumerate(tqdm(dataloader, desc=f"Attack {atk.name} for model {model.name}"))
        else:
            progress_bar = enumerate(dataloader)

        for idx, (batch, label, element_info) in progress_bar:
            batch = batch.to(device)
            label = label.to(device)
            y_one_hot = torch.nn.functional.one_hot(label, num_classes=num_classes)
            x_adv = atk.generate(
                x=batch,
                y=y_one_hot
            ).detach()

            with torch.no_grad():
                out = model(batch)
                out_adv = model(x_adv)
            y_pred_adv = out_adv.argmax(dim=-1)
            y_pred = out.argmax(dim=-1)
            # adapt metrics counting for reference or standard attack
            is_identity_atk = atk.__class__.__name__.replace("Attack", "").lower() == "identitybaseline"
            if is_identity_atk:
                y_pred = label
            else:
                mask = torch.eq(label, y_pred)
                label = label[mask]
                x_adv = x_adv[mask]
                batch = batch[mask]
                out = out[mask]
                out_adv = out_adv[mask]
                y_pred = y_pred[mask]
                y_pred_adv = y_pred_adv[mask]
            input_stat = {
                'x_adv': x_adv.detach(),
                'x': batch.detach(),
                'y': label,
                'out': out,
                'out_adv': out_adv,
                'y_pred': y_pred,
                'y_pred_adv': y_pred_adv 
            }
            statistics_composer.update(**input_stat)

        statistics_results = statistics_composer.compute()
        statistics_states = statistics_composer.get_raw_state()
        statistics_composer.reset()
        torch.cuda.empty_cache()
        return {"statistics": statistics_results, "statistics_states":statistics_states}

    def get_model_dataset_info(self) -> dict:
        batch, _, _ = next(iter(self.config.dataloader))
        return {
            'name': self.config.model.name,
            'parameters': sum([param.numel() for param in self.config.model.parameters()]),
            'classes': self.config.num_classes,
            'dimensionality': batch[0].shape
        }



    def evaluate_attacks(self) -> dict:
        """
        Evaluate model on the given dataset.
        In detail evaluate method will
        - Collect general information about dataset and model
        - Run attacks and obtain attack-specific evaluation


        Returns
        results = {
            'info': {},
            'attacks': {
                "atk_1": {"statistics":{}, "statistics_states":{}}
            }
        }
        """

        # extract the general information
        self.results['info'] = self.get_model_dataset_info()
        attack_evaluation_parameters = {}

        for i, attack_config in enumerate(self.config.attacks):
            attack_config_dict = {k:v for k,v in attack_config.model_dump().items() if v is not None}
            atk_id = attack_config_dict.get("id", attack_config_dict["name"])
            if atk_id in attack_evaluation_parameters:
                raise ValueError(f"{atk_id} is already setup for evaluation")
            attack_evaluation_parameters[atk_id] = dict(
                dataloader=self.config.dataloader,
                model=self.config.model,
                attack_config=attack_config_dict,
                statistics=self.config.statistics,
                device=self.config.device,
                num_classes=self.config.num_classes,
                verbose=True
            )

        # moved attack evaluation execution here
        self.results["attacks"] = {atk_id:self.evaluate_attack(**params) for atk_id, params in attack_evaluation_parameters.items()}
        return self.results
    
    def plan_attacks_evaluation(self) -> Plan:
        """
        Outputs an attack Plan to be executed by an Executor class
        """

        self.results['info'] = self.get_model_dataset_info()
        attack_evaluation_parameters = {}

        for i, attack_config in enumerate(self.config.attacks):
            attack_config_dict = {k:v for k,v in attack_config.model_dump().items() if v is not None}
            atk_id = attack_config_dict.get("id", attack_config_dict["name"])
            if atk_id in attack_evaluation_parameters:
                raise ValueError(f"{atk_id} is already setup for evaluation")
            attack_evaluation_parameters[atk_id] = dict(
                dataloader=self.config.dataloader,
                model=self.config.model,
                attack_config=attack_config_dict,
                statistics=self.config.statistics,
                device=self.config.device,
                num_classes=self.config.num_classes,
                verbose=True
            )

        worker_action = Evaluator.evaluate_attack_action
        for atk_id, worker_params in attack_evaluation_parameters.items():
            worker_params["atk_id"] = atk_id
            worker_params["dataset_name"] = self.config.dataloader.dataset.dataset.name
            worker_params["model_name"] = self.config.model.name
            worker_params["output_path"] = self.config.output_path
        
        action = Evaluator.save_info_to_disk
        params = dict(
            results_info=self.results["info"],
            dataset_name=self.config.dataloader.dataset.dataset.name,
            model_name=self.config.model.name,
            output_path=self.config.output_path
        )
        plan = Plan(
            worker_action=worker_action,
            worker_params=attack_evaluation_parameters,
            action=action,
            params=params
        )
        return plan

    def evaluate(self) -> dict:
        """This method is intended to provide the call method for evaluator class
        It provides in memory execution of all attacks and is inclusive of final aggregation.
        This reproduces the legacy behavior of Evaluator class, which executes all attacks and provides results as an object in memory

        self.results = {
            'info': {},  # general information about dataset and model
            'attacks': {},  # results of attacks evaluation
            'aggregate_statistics': {}  # Aggregate information from evaluated attacks if available (optional)
        }
        """
        _ = self.evaluate_attacks()
        statistics_composer = StatisticComposer(config=ConfigStatisticComposer(
            statistics=self.config.statistics,
            num_classes=self.results["info"]["classes"],
        ))
        statistics_composer.aggregator()
        self.results["aggregate_statistics"] = self.aggregate_attacks_statistics(
            statistics_composer=statistics_composer,
            results=self.results
        )
        return self.results


    @staticmethod
    def aggregate_attacks_statistics(statistics_composer: StatisticComposer, results: dict) -> dict:
        """Use statistic composer in aggregation mode to aggregate statistics states and 
        compute aggregated results

        """
        for attack, attack_results in results['attacks'].items():
            if not attack == "reference":
                statistics_composer.update_aggregate(attack_results["statistics_states"])
        aggregate_metrics = statistics_composer.compute()
        statistics_composer.reset()
        return aggregate_metrics

    def save_results_to_disk(self, output_path: str | pathlib.Path | None = None) -> None:
        """
        Save evaluation results to a JSON file
        self.results = dict(info, attacks=dict(attacks=dict(statistics, statistics_states)), aggregate_statistics:optional)
        """
        if not output_path:
            model_result_path = Path(self.config.output_path) / self.config.dataloader.dataset.dataset.name / self.config.model.name
        else:
            model_result_path = output_path
        os.makedirs(model_result_path, exist_ok=True)
        with open(model_result_path / "info.json", 'w') as f:
            json.dump(self.results["info"], f)
        for atk_id, atk_res in self.results["attacks"].items():
            attack_result_path = model_result_path / atk_id
            os.makedirs(attack_result_path, exist_ok=True)
            with open(attack_result_path / "statistics.json", 'w') as f:
                json.dump(atk_res["statistics"], f)
            with open(attack_result_path / "statistics_states.pkl", 'wb') as f:
                pickle.dump(atk_res["statistics_states"], f)
        if self.results["aggregate_statistics"]:
            with open(model_result_path / "aggregate_statistics.json", 'w') as f:
                json.dump(self.results["aggregate_statistics"], f)
        logging.info(f"Results saved to {model_result_path}")

    @staticmethod
    def save_attack_result_to_disk(atk_result : dict,
                                   atk_id : str,
                                   dataset_name : str , 
                                   model_name : str , 
                                   output_path: str | pathlib.Path ) -> None:
        """
        Save single attack results from static method -evaluate_attack- to a JSON file
        """
        
        model_result_path = Path(output_path) / dataset_name / model_name
        os.makedirs(model_result_path, exist_ok=True)
        atk_res = atk_result
        attack_result_path = model_result_path / atk_id
        os.makedirs(attack_result_path, exist_ok=True)
        with open(attack_result_path / "statistics.json", 'w') as f:
            json.dump(atk_res["statistics"], f)
        with open(attack_result_path / "statistics_states.pkl", 'wb') as f:
            pickle.dump(atk_res["statistics_states"], f)
        
        logging.info(f"Single attack results saved to {model_result_path}")

    @staticmethod
    def save_info_to_disk(results_info : dict,
                          dataset_name : str , 
                          model_name : str , 
                          output_path: str | pathlib.Path ) -> None:
        """
        Save single info to a JSON file
        """
        
        
        model_result_path = Path(output_path) / dataset_name / model_name
        os.makedirs(model_result_path, exist_ok=True)
        with open(model_result_path / "info.json", 'w') as f:
            json.dump(results_info, f)
        logging.info(f"Info saved to {model_result_path}")

    @staticmethod
    def evaluate_attack_action(**kwargs):
        """
        This function is the worker action of Path object.
        """
        sig = signature(Evaluator.evaluate_attack)
        accepted_params = sig.parameters
        atk_params = {k: v for k, v in kwargs.items() if k in accepted_params}
        atk_result = Evaluator.evaluate_attack(**atk_params)

        sig = signature(Evaluator.save_attack_result_to_disk)
        accepted_params = sig.parameters
        save_params = {k: v for k, v in kwargs.items() if k in accepted_params}
        Evaluator.save_attack_result_to_disk(atk_result, **save_params)
        


    @staticmethod
    def read_results_from_disk(results_dir: str | pathlib.Path):
        """Read from disk back to Evaluator results object structure
        The directory from which results are read are the same kind of the target of `save_`
        self.results = dict(info, attacks=dict(attacks=dict(statistics, statistics_states)), aggregate_statistics:optional)
        """
        results_dir = Path(results_dir)
        results = {"attacks":{}}
        attacks_dir = [attack_dir for attack_dir in results_dir.iterdir() if attack_dir.is_dir()]
        for attack_dir in attacks_dir:
            with open(attack_dir / "statistics.json", "r") as fmetric:
                statistics = json.load(fmetric)
            with open(attack_dir / "statistics_states.pkl", "rb") as fdata:
                statistics_states = pickle.load(fdata)
            results["attacks"][attack_dir.name] = {
                "statistics": statistics,
                "statistics_states": statistics_states
            }
        if "aggregate_statistics.json" in results_dir.iterdir():
            with open(results_dir / "aggregate_statistics.json", "r") as f:
                results["aggregate_statistics"] = json.load(f)
        if "info.json" in results_dir.iterdir():
            with open(results_dir / "info.json", "r") as f:
                results["info"] = json.load(f)
        return results

    def __repr__(self):
        return f'Evaluator(dataset={self.config.dataloader.dataset.dataset.name}, model={self.config.model.name})'






