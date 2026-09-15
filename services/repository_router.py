import json
from pathlib import Path
from typing import Literal, Union, Annotated

from fastapi import APIRouter, Query, Depends, Request, Body, HTTPException
from pydantic import BaseModel, Field

from models import config_field, ModelReportProps, DatasetReportProps, ModelInfo, DatasetInfo

router = APIRouter(prefix="/repository")


def get_path(request: Request, repo_path: str = Query(...)):
    config_func = config_field(attr_name=repo_path)
    return config_func(request)


# Define the discriminated union
InfoUnion = Annotated[
    Union[DatasetInfo, ModelInfo, ModelReportProps, DatasetReportProps],
    Field(discriminator='type')
]

_MODEL_MAP = {
    "model": ModelInfo,
    "dataset": DatasetInfo,
    "report_model": ModelReportProps,
    "report_dataset": DatasetReportProps,
}


def _extract_task(model_type: str, info) -> str:
    if model_type in ["report_model", "report_dataset"]:
        return info.info.task
    return info.task


@router.get(
    "/getList",
    response_model=Union[
        list[ModelInfo],
        list[DatasetInfo],
        list[ModelReportProps],
        list[DatasetReportProps]
    ]
)
def get_info(
        tasks: list[str] | None = Query(
            default=None,
            description="Task(s) to filter the reports with."
        ),
        repo_path: str | Path = Depends(get_path),
        model_type: Literal["model", "dataset", "report_model", "report_dataset"] = Query(
            default="model",
            description="Type of item to filter the reports with."
        ),
) -> list[ModelInfo] | list[DatasetInfo] | list[ModelReportProps] | list[DatasetReportProps]:
    """
    Get all model/datasets/reports under `repo_path` matching `tasks`.
    """
    if isinstance(repo_path, str):
        repo_path: Path = Path(repo_path).expanduser()
    if not repo_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Repository path '{repo_path}' does not exist."
        )

    task_filter = set(tasks) if tasks else None
    print(f"model type = {model_type}")
    model_cls = _MODEL_MAP[model_type]

    out = []
    file_name: str = "report.json" if model_type == "report_model" else "info.json"
    for full_path in repo_path.glob(f"**/{file_name}"):
        root = full_path.parent
        print(full_path)
        with full_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)

        raw["repository"] = str(root)
        if model_type in ("model", "dataset") and not raw.get("id"):
            raw["id"] = root.name
        try:
            item = model_cls.model_validate(raw)

            task = _extract_task(model_type, item)
            if task_filter is None or task in task_filter:
                out.append(item)
        except:
            print(f"Cannot load info.json from {root}")
            continue
    return out
