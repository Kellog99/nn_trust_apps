from typing import Optional, List, Literal

from pydantic import BaseModel, ConfigDict

from attack_server.models.info import ModelInfo, DatasetInfo


class ReportMetricsProps(BaseModel):
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    f1score: Optional[float] = None
    confusion_matrix: Optional[List[List[int | float]]] = None
    robustness: Optional[float] = None
    wobbliness: Optional[float] = None


class ReportAttacksProps(BaseModel):
    name: str
    risk: float
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    f1score: Optional[float] = None
    misclassification: Optional[float] = None
    power: Optional[float] = None
    num_queries: Optional[int] = None
    robustness: Optional[float] = None
    confusion_matrix: Optional[List[List[int]]] = None


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
    attacks: dict[str, ReportAttacksProps]
