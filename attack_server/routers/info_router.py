import logging

import torch
from fastapi import APIRouter, HTTPException, Request, Depends, Body

from attack_server.lib.model import RegisteredObject
from attack_server.models.main_model import ServerConfig, SharableVariables, config_field
from attack_server.utils.utils import get_parameter_prop
from nn_trust.attack.attack_factory import EvasionAttackFactory as EAF, AttackInfo
from nn_trust.core import Task
from nn_trust.evaluation.statistic_factory import StatisticsFactory as SF, InfoStatistic

router = APIRouter(prefix="/info")


@router.get("/attacks")
def get_attacks_info(
        excluded_attacks: list[str] = Depends(config_field(attr_name="excluded_attacks"))
) -> dict[str, RegisteredObject]:
    """
    Get the list of all the available attacks for a specific task.
    """
    out: dict[str, RegisteredObject] = {}
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
def get_statistics_info(
        excluded_statistics: list[str] = Depends(config_field(attr_name="excluded_statistics"))
) -> dict[str, RegisteredObject]:
    """
    Get the list of all the available statistics that can be measured during a benchmark
    """
    try:

        out: dict[str, RegisteredObject] = {}

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
def get_variables_info(
        config: ServerConfig = Depends(config_field(attr_name=None))
) -> SharableVariables:
    """
    This function handles the sharing of the backend variables with the frontend.

    Args:
        config:
    """
    # These are all the variables that have been set so far
    return SharableVariables.model_validate(config.model_dump())


@router.post("/saveConfiguration")
def save_configuration(
        request: Request,
        new_config: dict = Body(...)
) -> ServerConfig:
    """
    This function handles the saving of the  new configuration file after the modifications on the frontend.
    In this case, it is impossible to avoid the `Request` in the input because it is its state that has to be changed

    Args:
        request:  fastapi state
        new_config: New configuration file

    Returns:

    """
    config: ServerConfig = request.app.state.config
    for key, value in new_config.items():
        setattr(config, key, value)
    request.app.state.config = config
    return request.app.state.config
