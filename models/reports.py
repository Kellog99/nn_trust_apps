from typing import Optional, List, Any, Literal

from pydantic import BaseModel, ConfigDict

from models.info import ModelInfo, DatasetInfo
from models.model import ParametersProps


# These are the metrics associated to the model's performance
class ReportMetricsProps(BaseModel):
    num_samples: Optional[int] = None
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    f1score: Optional[float] = None
    confusion_matrix: Optional[List[List[int | float]]] = None
    robustness: Optional[float] = None
    wobbliness: Optional[float] = None


# These are the metrics that compute the attack's performance
class AttackMetricsProps(BaseModel):
    risk: Optional[float] = None
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    f1score: Optional[float] = None
    misclassification: Optional[float] = None
    power: Optional[float] = None
    num_queries: Optional[int] = None
    robustness: Optional[float] = None
    confusion_matrix: Optional[List[List[int]]] = None


class ParameterLog(BaseModel):
    """
    This base model represents the log of a parameter in a specific format.
    """
    id: str
    name: Optional[str] = None
    value: Any
    description: Optional[str] = None


class ReportAttackProps(BaseModel):
    name: str
    metrics: AttackMetricsProps
    parameters: list[ParameterLog]


class ReportProps(BaseModel):
    model_config = ConfigDict(extra='allow')


############# Dataset #############
class DatasetReportProps(ReportProps):
    type: Literal["dataset_report"] = "dataset_report"
    info: DatasetInfo


############## Model ##############
class ModelReportProps(ReportProps):
    type: Literal["model_report"] = "model_report"
    info: ModelInfo
    metrics: ReportMetricsProps
    attacks: dict[str, ReportAttackProps]
