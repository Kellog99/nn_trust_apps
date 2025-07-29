import argparse
import os.path
import pathlib
from pathlib import Path
from typing import Optional

import torch.nn
import torchvision.transforms
import yaml

from .evaluator import EvaluatorConfig
from .utils import get_dataloader, get_model


class StoreTrueIfSet(argparse.Action):
    def __init__(self, option_strings, dest, nargs=0, **kwargs):
        super().__init__(option_strings, dest, nargs=nargs, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, True)


abbrev = []  # list of all the abbreviation for showing in the Help command


def get_abbreviation(name) -> str:
    i = 0
    abr = name[:i]
    while abr in abbrev:
        i += 1
        abr = name[:i]
    abbrev.append(abr)
    return abr


def get_parser(model_fields):
    """
    Using the Pydantic model for creating the associated argparser.
    """

    parser = argparse.ArgumentParser()
    for name, field in model_fields.items():
        abr = get_abbreviation(name)
        hlp = field.description
        if field.default is not None:
            hlp = hlp + f" Default {field.default}."
        arguments = {"dest": name, "help": hlp}
        if field.annotation is bool:
            arguments["action"] = StoreTrueIfSet

        if field.annotation in [int, str, float]:
            arguments["type"] = field.annotation

        elif field.annotation is list:
            arguments["nargs"] = "*"

        parser.add_argument(f"--{name}", f"-{abr}", **arguments)
    return parser.parse_args()


def read_config_file(config_filename: str) -> dict:
    """
    Read the configuration file and return the content as a dictionary.
    """
    if not os.path.exists(config_filename):
        raise ValueError(f"File not found: {config_filename}")

    if config_filename.endswith((".yaml", ".yml")):
        with open(config_filename, "r") as f:
            config_data = yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported file format: {config_filename}")
    return config_data


def get_data_transformation_config(
    transform_id: str,
    size: int,
    crop: Optional[int] = None,
    # mean: Optional[list[float]] = None,
    # std: Optional[list[float]] = None
):
    """
    An utility function providing the transformation and inverse transformation
    for dataset at hand"""

    if transform_id == "imagenet":
        # mean, std = [0.5074, 0.5308, 0.5306], [0.2639, 0.2518, 0.2521]
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize(size=(size, size)),
                torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                # torchvision.transforms.Normalize(mean=[0.5074, 0.5308, 0.5306], std=[0.2639, 0.2518, 0.2521])
            ]
        )
    elif transform_id == "imagenet_like_crop":
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize(size=(size, size)),
                torchvision.transforms.CenterCrop(crop),
                torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
    else:
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize(size=(size, size)),
                torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    if transform_id == "imagenet":
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        inverse_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Normalize(
                    mean=[-mean_el / std_el for mean_el, std_el in zip(mean, std)], std=[1 / std_el for std_el in std]
                ),
                torchvision.transforms.Resize(size=(size, size)),
                torchvision.transforms.ToPILImage(),
            ]
        )
    if transform_id == "imagenet_like_crop":
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        inverse_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Normalize(
                    mean=[-mean_el / std_el for mean_el, std_el in zip(mean, std)], std=[1 / std_el for std_el in std]
                ),
                torchvision.transforms.Resize(size=(size, size)),
                torchvision.transforms.ToPILImage(),
            ]
        )
    else:
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        inverse_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Normalize(
                    mean=[-mean_el / std_el for mean_el, std_el in zip(mean, std)], std=[1 / std_el for std_el in std]
                ),
                torchvision.transforms.Resize(size=(size, size)),
                torchvision.transforms.ToPILImage(),
            ]
        )

    return transform, inverse_transform


def get_config(config_filename: str) -> EvaluatorConfig:
    """
    Define the arguments to insert in the command line.
    To do so, it is used a `Pydantic` base model to check the type of all the elements.
    """
    args = get_parser(model_fields=EvaluatorConfig.model_fields)

    config_path = args.config if args.config is not None else config_filename

    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            # The YAML file format is unusual with [evaluation] section header
            # We need to handle this custom format
            config = yaml.safe_load(f)
    else:
        raise ValueError("The path to the configuration file does not lead to anything.")

    # Sync all value:
    # command_line + yaml.
    configuration_dict = {}
    for key, value in config.items():
        if key == "attack_configurations":
            data["attack_configurations"] = value
        else:
            data = data | value

    arguments = vars(args)
    for key, value in arguments.items():
        if value is not None:
            data[key] = value

    for var in ["batch", "subset", "type_dataset", "num_workers", "mean", "std"]:
        # These are all defaults values that are needed for getting the dataloader
        if var not in data:
            # TODO remove reference to Evaluator config, config reading function should not depend on EvaluatorConfig
            # It should just read a config file or dictionary, other classes will set default if needed by them
            data[var] = EvaluatorConfig.model_fields[var].default

    data["dataloader"] = get_dataloader(**data)

    # TODO remove hardcoded values, link to dataset and model selection
    data["inverse_transformation"] = torchvision.transforms.Compose(
        [
            torchvision.transforms.Normalize(
                mean=[-mean / std for mean, std in zip(data["mean"], data["std"])], std=[1 / std for std in data["std"]]
            ),
            torchvision.transforms.Resize(size=(224, 224)),
            torchvision.transforms.ToPILImage(),
        ]
    )

    if isinstance(data["model"], str):
        if os.path.isfile(data["model"]):
            model_name = str(Path(data["model"]).name)
        else:
            model_name = data["model"]
    elif isinstance(data["model"], torch.nn.Module):
        model_name = data["model"]._get_name()

    result_file = os.path.join(data["out"], model_name, "data.json")
    if os.path.exists(result_file) and data["load_results"]:
        with open(result_file, "r") as f:
            # The YAML file format is unusual with [evaluation] section header
            # We need to handle this custom format
            results = yaml.safe_load(f)

        if "atk" in results.keys():
            data["attacks"] = [atk for atk in data["attacks"] if atk not in results["atk"]]
        else:
            raise ValueError("The results that have been loaded do not provide any list of attacks.")

    return data


if __name__ == "__main__":
    config = get_config()
    print(str(config))
    print(config.inverse_transformation)
