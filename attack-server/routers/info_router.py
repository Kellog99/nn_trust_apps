import logging
from fastapi import APIRouter
from annotated_types import Gt, Ge, Le, Lt
from fastapi import HTTPException
from nn_trust.attack._evasion import EvasionAttackFactory as EAF
from nn_trust.attack.evaluation._statistics import StatisticsFactory as SF
from nn_trust.core import Task
from pydantic_core import PydanticUndefined
from typing import Optional
from pydantic import BaseModel

class ParametersProps(BaseModel):
    name: str
    label: str
    min: float
    max: float
    step: float
    default: float
    description: str

class AttackProps(BaseModel):
    id: str
    name: str
    knowledge: str
    type: str
    description: Optional[str] = None
    parameters: list[ParametersProps]

class AttackPost(BaseModel):
    title: str
    body: str
    id: float

class MetricProps(BaseModel):
    id: str
    name: str
    description: str


router = APIRouter(prefix="/info")


@router.get("/attacks/getInfo")
def get_attacks_info():
    """
    Get the list of all the available attacks for a specific task.
    """
    try:
        out = {}
        knowledge = {
            0: "White",
            1: "Black"
        }

        type = {
            0: "Physical",
            1: "Digital"
        }
        for atk in EAF.list_attacks(name=False):
            if hasattr(atk, "_name"):
                atk_id = atk.__name__.removesuffix("Attack").lower()
                # collecting all the parameters for displaying the configuration
                parameters = []
                for param_name, param_info in EAF.list_config_param(atk_id, (int, float)):
                    max_value = 1000
                    min_value = 1
                    if len(param_info.metadata) > 0:
                        # Extracting from the metadata the maximum value and minimum value of the parameters
                        for val in param_info.metadata:
                            if isinstance(val, (Gt, Ge)):
                                atr = getattr(val, 'ge' if isinstance(val, Ge) else 'gt')
                                if atr != -float('inf'):
                                    min_value = getattr(val, 'ge' if isinstance(val, Ge) else 'gt')
                                else:
                                    min_value = -10000
                            elif isinstance(val, (Lt, Le)):
                                atr = getattr(val, 'le' if isinstance(val, Le) else 'lt')
                                if atr != float('inf'):
                                    max_value = atr
                    if param_info.default is PydanticUndefined:
                        default = min_value
                    else:
                        default = param_info.default if param_info.default != float('inf') else max_value
                    if hasattr(param_info, 'step'):
                        step = getattr(param_info, 'step')
                    else:
                        step = (max_value - min_value) / 1000
                    parameters.append(
                        ParametersProps(
                            name=param_name,
                            label=param_name,
                            min=min_value,
                            max=max_value,
                            step=step,
                            default=default,
                            description=param_info.description
                        ))

                # Creating the list of all the attacks

                out[atk_id] = AttackProps(
                    id=atk_id,
                    name=atk._name,
                    knowledge=knowledge[atk.ATTACK_KNOWLEDGE.value],
                    description=atk._description if hasattr(atk,
                                                            "_description") else "this should be a description about this particular attack. However we have not being able to add that.",
                    type=type[atk.ATTACK_TYPE.value],
                    parameters=parameters
                )
            else:
                print(atk)
        return out

    except Exception as e:
        logging.error(f"Unexpected error during get result: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error during get result {str(e)}")

@router.get("/metrics/getInfo")
def get_statistics_info() -> dict[str, MetricProps]:
    """
    Get the list of all the available statistics that can be measured during a benchmark
    """
    try:
        return {
            stat: MetricProps(
                id=stat,
                name=stat,
                description="-"
            ) for stat in SF.list_statistics(name=False, task=Task.Classification)
        }

    except Exception as e:
        logging.error(f"Unexpected error during get result: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error during get result {str(e)}")