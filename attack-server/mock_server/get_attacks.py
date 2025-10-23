from annotated_types import Gt, Ge, Le, Lt
from nn_trust.attack._evasion import EvasionAttackFactory as EAF
from nn_trust.core import Task
from pydantic_core import PydanticUndefined

from models import ParametersProps, AttackProps


def get_attacks() -> dict:
    """
    Return the list of all the available attacks for a specific task.
    """
    out = {}
    for atk in EAF.list_attacks(name=False, task=[Task.Classification]):
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
            atk_props = AttackProps(
                id=atk_id,
                name=atk._name,
                knowledge=atk.ATTACK_KNOWLEDGE,
                description=atk._description if hasattr(atk,
                                                        "_description") else "this should be a description about this particular attack. However we have not being able to add that.",
                type=atk.ATTACK_TYPE,
                parameters=parameters
            )
            out[atk_id] = atk_props.model_dump()
        else:
            print(atk)
    return out
