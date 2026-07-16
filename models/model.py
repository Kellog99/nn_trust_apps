from typing import Dict, Literal, Optional

from pydantic import BaseModel


########################### Register Object ###########################
class ParametersProps(BaseModel):
    id: str
    name: str
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    default: float | int | str | bool
    description: Optional[str] = None
    kind: Optional[Literal["number", "enum"]] = "number"
    options: Optional[list[str]] = None


class RegisteredObject(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    parameters: list[ParametersProps]
    task: str
    knowledge: Optional[str] = None
    objective: Optional[str] = None
    privacy_type: Optional[str] = None


class BenchmarkModelProps(BaseModel):
    name: str
    param: int
    task: str
    benchmark_id: str
    metrics: Dict[str, float | int]
