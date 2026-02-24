import logging

import torch
from fastapi import APIRouter, HTTPException, Request

from lib.model import RegisteredObject
from models.main_model import ServerConfig, SharableVariables
from nn_trust.attack.attack_factory import EvasionAttackFactory as EAF, AttackInfo
from nn_trust.core import Task
from nn_trust.evaluation.statistic_factory import StatisticsFactory as SF, InfoStatistic
from utils.utils import get_parameter_prop

router = APIRouter(prefix="/info")


@router.get("/attacks")
def get_attacks_info(request: Request) -> dict[str, RegisteredObject]:
    """
    Get the list of all the available attacks for a specific task.
    """

    out: dict[str, RegisteredObject] = {}
    excluded_attacks: list[str] = request.app.state.config.excluded_attacks
    for atk in EAF.get_list_classes(task={Task.Classification}):
        if atk in excluded_attacks:
            continue
        atk_info: AttackInfo = AttackInfo.model_validate(EAF.get_information(id=atk, exclude=set()))

        # collecting all the parameters for displaying the configuration
        parameters = []

        for param_id, param_info in EAF.get_config_param(atk, (int, float)):
            parameters.append(get_parameter_prop(id=param_id, param_info=param_info))

        # Creating the list of all the attacks
        out[atk] = RegisteredObject(
            id=atk_info.id,
            name=atk_info.name,
            task=Task.Classification.name,
            knowledge=atk_info.knowledge.name,
            description=atk_info.description,
            parameters=parameters
        )
    return out


@router.get("/metrics")
def get_statistics_info(request: Request) -> dict[str, RegisteredObject]:
    """
    Get the list of all the available statistics that can be measured during a benchmark
    """
    try:

        out: dict[str, RegisteredObject] = {}
        excluded_statistics: list[str] = request.app.state.config.excluded_statistics

        for stat in SF.get_list_classes(task={Task.Classification}):
            if stat in excluded_statistics:
                continue
            metric_info: InfoStatistic = InfoStatistic.model_validate(SF.get_information(id=stat, exclude=set()))

            # collecting all the parameters for displaying the configuration
            parameters = []
            for param_id, param_info in SF.get_config_param(stat, (int, float)):
                try:
                    parameters.append(get_parameter_prop(
                        id=param_id,
                        param_info=param_info
                    ))
                except:
                    print(f"unable to save the {param_id} parameters.")

            # Creating the list of all the attacks
            out[stat] = RegisteredObject(
                id=metric_info.id,
                name=metric_info.name,
                task=Task.Classification.name,
                description=metric_info.description,
                parameters=parameters
            )
        return out

    except Exception as e:
        logging.error(f"Unexpected error during get result: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error during get result {str(e)}")


@router.get("/devices")
def get_devices() -> list[str]:
    """
    Returns a list of all available compute devices (CPU, CUDA, MPS).

    Returns:
        list: A list of all the available devices
    """
    devices: list[str] = ["cpu"]

    # Check for CUDA devices
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        for i in range(device_count):
            devices.append(torch.cuda.get_device_name(i))

    # Check for MPS (Apple Silicon)
    if torch.backends.mps.is_available():
        devices.append("mps")

    return devices


@router.get("/variables")
def get_variables_info(request: Request) -> SharableVariables:
    # These are all the variables that have been set so far
    out: ServerConfig = request.app.state.config
    return SharableVariables.model_validate(out.model_dump())


@router.post("/saveConfiguration")
def save_configuration(new_config: dict, request: Request) -> ServerConfig:
    request.app.state.config = ServerConfig(**new_config)
    return request.app.state.config
