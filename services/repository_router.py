import json
import os
from pathlib import Path
from pprint import pprint
from typing import Literal, Union, Annotated

from fastapi import APIRouter, Query, Depends, Request, Body, HTTPException
from pydantic import BaseModel, Field, TypeAdapter

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
    for root, _, files in os.walk(repo_path):
        for file in files:
            # The goal is to extract only those files that are a json file
            if file != "info.json":
                continue
            full_path: Path = Path(root) / file
            with open(full_path, "r", encoding="utf-8") as f:
                raw = json.load(f)

            raw["repository"] = root
            if model_type in ("model", "dataset") and not raw.get("id"):
                raw["id"] = Path(root).name
            try:
                item = model_cls.model_validate(raw)

                task = _extract_task(model_type, item)
                if task_filter is None or task in task_filter:
                    out.append(item)
            except:
                print(f"Cannot load info.json from {root}")
                continue
    return out


@router.post("/upload")
def upload(
        file: dict = Body(...),
        repo_path: str | Path | None = Depends(get_path),
):
    """
    Upload a .zip file and organize it.

    Args:
        file: zip file
        repo_path: repository folder where the file has to be uploaded

    """
    if repo_path:
        if isinstance(repo_path, (str, Path)):
            if isinstance(repo_path, str):
                repo_path: Path = Path(repo_path).expanduser()
        else:
            raise ValueError("The type of the path is not supported.")
    else:
        repo_path = Path("~/Desktop/StableAI").expanduser()
    repo_path.mkdir(parents=True, exist_ok=True)
    base_model: BaseModel | None = None
    for model in [DatasetInfo, ModelInfo, ModelReportProps, DatasetReportProps]:
        try:
            base_model = model.model_validate(file)
            break
        except:
            print(f"Exclusion of the model {model}")
    return {}
