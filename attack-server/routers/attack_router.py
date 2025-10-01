from fastapi import APIRouter 
from typing import List
from annotated_types import Gt, Ge, Le, Lt
from nn_trust.attack._evasion import EvasionAttackFactory as EAF
from nn_trust.attack.evaluation._statistics import StatisticsFactory as SF
from nn_trust.core import Task
from pydantic_core import PydanticUndefined

# ---------------- Utilities --------------- #

"""
import here every function that you need in the services
"""

#from lib.attack_utils import ecc...

# ---------------- Services ---------------- #

router = APIRouter(prefix="/attack", tags=["attacks and statistics"])