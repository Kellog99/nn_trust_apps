import logging

from fastapi import APIRouter, HTTPException
from nn_trust.attack.attack_factory import EvasionAttackFactory as EAF, AttackInfo
from nn_trust.core import Task
from nn_trust.evaluation.statistic_factory import StatisticsFactory as SF

from lib.model import RegisteredObject
from .utils import get_parameter_prop

router = APIRouter(prefix="/info")


@router.get("/attacks/getInfo")
def get_attacks_info() -> dict[str, RegisteredObject]:
    """
    Get the list of all the available attacks for a specific task.
    """

    out: dict[str, RegisteredObject] = {}
    for atk in EAF.get_list_classes(task={Task.Classification}):

        atk_info: AttackInfo = EAF.get_information(id=atk, exclude={})

        # collecting all the parameters for displaying the configuration
        parameters = []

        for param_id, param_info in EAF.get_config_param(atk, (int, float)):
            parameters.append(get_parameter_prop(id=param_id, param_info=param_info))

        # Creating the list of all the attacks
        out[atk] = RegisteredObject(
            id=atk_info['id'],
            name=atk_info['name'],
            task=Task.Classification.name,
            knowledge=atk_info['knowledge'].name,
            description=atk_info['description'],
            parameters=parameters
        )
    return out


@router.get("/metrics/getInfo")
def get_statistics_info() -> dict[str, RegisteredObject]:
    """
    Get the list of all the available statistics that can be measured during a benchmark
    """
    try:
        out: dict[str, RegisteredObject] = {}
        for stat in SF.get_list_classes(task={Task.Classification}):

            metric_info: AttackInfo = SF.get_information(id=stat, exclude={})

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
                id=metric_info['id'],
                name=metric_info['name'],
                task=Task.Classification.name,
                description=metric_info['description'],
                parameters=parameters
            )
        return out

    except Exception as e:
        logging.error(f"Unexpected error during get result: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error during get result {str(e)}")
