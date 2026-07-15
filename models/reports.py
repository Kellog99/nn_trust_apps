from typing import Optional, List

from pydantic import BaseModel, ConfigDict

from models.info import ModelInfo, DatasetInfo
from models.model import ParametersProps


# These are the metrics associated to the model's performance
class ReportMetricsProps(BaseModel):
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    f1score: Optional[float] = None
    confusion_matrix: Optional[List[List[int | float]]] = None
    robustness: Optional[float] = None
    wobbliness: Optional[float] = None


# These are the metrics that compute the attack's performance
class AttackMetricsProps(BaseModel):
    risk: float
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    f1score: Optional[float] = None
    misclassification: Optional[float] = None
    power: Optional[float] = None
    num_queries: Optional[int] = None
    robustness: Optional[float] = None
    confusion_matrix: Optional[List[List[int]]] = None


class ReportAttackProps(BaseModel):
    name: str
    metrics: AttackMetricsProps
    parameters: list[ParametersProps]


class ReportProps(BaseModel):
    model_config = ConfigDict(extra='allow')


############# Dataset #############
class DatasetReportProps(ReportProps):
    info: DatasetInfo


############## Model ##############
class ModelReportProps(ReportProps):
    info: ModelInfo
    metrics: ReportMetricsProps
    attacks: dict[str, ReportAttackProps]
