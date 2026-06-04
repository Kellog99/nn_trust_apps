import logging
from typing import Literal, get_origin

import torch
from fastapi import APIRouter, HTTPException, Request, Depends, Body

from attack_server.lib.model import RegisteredObject
from attack_server.models.main_model import ServerConfig, SharableVariables, config_field
from attack_server.utils.utils import get_parameter_prop
from benchmarking.privacy.contracts import get_privacy_dataset_factory
from benchmarking.privacy.model_registry import AppPrivacyModelFactory
from nn_trust.attack.utils.model_building import get_default_model_factory
from nn_trust.attack.attack_factory import AttackFactory, AttackInfo
from nn_trust.core import Task
from nn_trust.evaluation.statistic_factory import StatisticsFactory as SF, InfoStatistic

router = APIRouter(prefix="/info")


def _str_enum(v) -> str | None:
    """Safely stringify an enum value or return None."""
    if v is None:
        return None
    return str(v.value) if hasattr(v, "value") else str(v)


def _collect_params(atk: str) -> list:
    params = []
    seen: set[str] = set()
    for type_filter in ((int, float, str), None):
        for pid, pinfo in AttackFactory.get_config_param(atk, type_filter):
            if pid in seen:
                continue
            seen.add(pid)
            if type_filter is None and get_origin(pinfo.annotation) is not Literal:
                continue
            params.append(get_parameter_prop(id=pid, param_info=pinfo))
    return params


@router.get("/attacks")
def get_attacks_info(
    excluded_attacks: list[str] = Depends(config_field(attr_name="excluded_attacks")),
) -> dict[str, RegisteredObject]:
    """List all available attacks for classification."""
    out: dict[str, RegisteredObject] = {}
    for atk in AttackFactory.get_list_classes(task={Task.Classification}):
        if atk in excluded_attacks:
            continue
        info = AttackInfo.model_validate(AttackFactory.get_information(id=atk, exclude=set()))
        out[atk] = RegisteredObject(
            id=info.id,
            name=info.name,
            task=Task.Classification.name,
            knowledge=info.knowledge.name if info.knowledge else None,
            description=info.description,
            parameters=_collect_params(atk),
            objective=_str_enum(getattr(info, "objective", None)),
            privacy_type=_str_enum(getattr(info, "privacy_type", None)),
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


@router.get("/privacy/datasets")
def get_privacy_datasets() -> list[dict]:
    return [spec.info() for spec in get_privacy_dataset_factory().list_specs()]


@router.get("/privacy/models")
def get_privacy_models() -> list[dict]:
    factory = get_default_model_factory()
    if isinstance(factory, AppPrivacyModelFactory):
        return [spec.info() for spec in factory.list_specs()]
    return []  # fallback if factory not yet configured


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
