from typing import List

from annotated_types import Gt, Ge, Le, Lt
from nn_trust.attack._evasion import EvasionAttackFactory as EAF
from nn_trust.attack.evaluation._statistics import StatisticsFactory as SF
from nn_trust.core import Task
from pydantic_core import PydanticUndefined

from models import AttackProps, ParametersProps, MetricProps


def get_attacks(task: Task = Task.Classification) -> List[dict]:
    """
    Return the list of all the available attacks for a specific task.
    """
    out = []
    for atk in EAF.list_attacks(task=task, name=False):
        atk_id = atk.__name__.removesuffix("Attack").lower()
        print(atk_id)

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
        atk_props = AttackProps(
            id=atk_id,
            name=atk_id,
            knowledge=atk.ATTACK_KNOWLEDGE,
            type=atk.ATTACK_TYPE,
            parameters=parameters
        )
        out.append(atk_props.model_dump())
    return out


def get_metrics() -> List[dict]:
    """
    Return the list of all the available metrics that are computable during a benchmark.
    """
    out = []
    for stat in SF.list_statistics(name=False):
        stat_props = MetricProps(
            id=stat.__name__.removesuffix("statistics").lower(),
            name=stat._name
        )
        out.append(stat_props.model_dump())

    return out
