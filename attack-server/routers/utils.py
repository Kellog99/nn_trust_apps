import math

from annotated_types import Gt, Ge, Le, Lt
from nn_trust.attack.attack_factory import EvasionAttackFactory as EAF, AttackInfo
from nn_trust.core import Task
from nn_trust.evaluation.statistic_factory import StatisticsFactory as SF
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

from lib.models import ParametersProps


def get_parameter_prop(id: str, param_info: FieldInfo) -> ParametersProps:
    """
    This function allows to properly produce a P
    """
    max_value = 1000
    min_value = 1

    if len(param_info.metadata) > 0:
        # Extracting from the metadata the maximum value and minimum value of the parameters
        # If there are no constraints than the max value and the min value are the one above indicated
        for val in param_info.metadata:
            if isinstance(val, (Gt, Ge)):
                min_value = getattr(val, 'ge' if isinstance(val, Ge) else 'gt')
            elif isinstance(val, (Lt, Le)):
                max_value = getattr(val, 'le' if isinstance(val, Le) else 'lt')

    # Handle infinity values - replace with reasonable defaults
    if math.isinf(max_value) or max_value > 1e10:
        max_value = 1000
    if math.isinf(min_value) or min_value < -1e10:
        min_value = 0 if param_info.annotation == float else 1

    # Ensure min < max
    if min_value > max_value:
        tmp = min_value
        min_value = max_value
        max_value = tmp

    # The default value, if not assigned, is the mean of the interval
    if param_info.default is PydanticUndefined:
        default = (max_value + min_value) / 2
    else:
        default = param_info.default
        # Clamp default to valid range
        default = max(min_value, min(max_value, default))

    if hasattr(param_info, 'step'):
        step = getattr(param_info, 'step')
    else:
        step = (max_value - min_value) / 1000
        if isinstance(param_info.annotation, int) or param_info.annotation == int:
            step = max(int(step), 1)

    name = getattr(param_info, "title") if hasattr(param_info, "title") and getattr(param_info, "title") != None else id
    return ParametersProps(
        name=name,
        label=id,
        min=float(min_value),
        max=float(max_value),
        step=float(step),
        default=float(default),
        description=param_info.description
    )
