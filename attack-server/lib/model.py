from typing import Dict, Any
from typing import Optional

from pydantic import BaseModel


########################### Register Object ###########################
class ParametersProps(BaseModel):
    id: str
    name: str
    min: float
    max: float
    step: float
    default: float
    description: str


class RegisteredObject(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    parameters: list[ParametersProps]
    task: str
    knowledge: Optional[str] = None


#######################################################################
class ExecutionConfig(BaseModel):
    dataset: str
    model: str
    attacks: Any
    metrics: Any


class BenchmarkModelProps(BaseModel):
    name: str
    param: int
    task: str
    benchmark_id: str
    metrics: Dict[str, float | int]
