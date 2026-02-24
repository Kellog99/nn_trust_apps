"""
This router handles all the repositories:
    * model
    * dataset
    * model_report
    * dataset_report
"""
import json
import os
from pathlib import Path
from typing import Literal, Union

from fastapi import APIRouter, Query, HTTPException, Depends, Request
from pydantic import BaseModel

from models.info import ModelInfo, DatasetInfo
from models.main_model import config_field
from models.reports import ModelReportProps, DatasetReportProps

router = APIRouter(prefix="/repository")


def get_path(request: Request, repo_path: str = Query(...)):
    config_func = config_field(attr_name=repo_path)
    return config_func(request)


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
        tasks: str | list[str] | None = Query(
            default=None,
            description="It represents the task to filter the reports with."
        ),
        file_checker: Literal["info.json", "report.json"] = Query(
            default=...,
            description="It tells what file has to be check and load inside the repository."
        ),
        repo_path: str | Path = Depends(get_path),
        base_model: type[BaseModel] = None
):
    """
    Get all models or dataset that are saved in the `repo_path` that satisfy the filter given by the `tasks`.
    It follows the general scaffolding that was defining in the beginning.

    Args:
        tasks (str | list[str] | None )
        file_checker
        repo_path
        base_model
    """

    ################### Validating the path ###################
    if repo_path:
        if isinstance(repo_path, str):
            repo_path: Path = Path(repo_path).expanduser()

        if not repo_path.exists():
            # Creating the folder if it does not exist.
            repo_path.mkdir(parents=True, exist_ok=True)
    else:
        raise ValueError("it is necessary to define the repository path.")
    ###########################################################

    out = []
    if tasks and isinstance(tasks, str):
        tasks = [tasks]

    # Walk through all subdirectories
    # and validate the json that it is found
    for root, dirs, files in os.walk(repo_path):
        if file_checker in files:
            file_info_path = os.path.join(root, file_checker)
            with open(file_info_path, "r") as f:
                info_json = json.load(f)
            # Dynamically saving all the absolute path for the file
            info_json["repository"] = root

            ############################ Instance of the base model ############################
            if base_model is None:
                for model in [DatasetInfo, ModelInfo, ModelReportProps, DatasetReportProps]:
                    try:
                        model.model_validate(info_json)
                        base_model = model
                        break
                    except:
                        print(f"Exclusion of the model {model}")
                if base_model is None:
                    raise ValueError("No model among the possible models is valid. Please retry.")
            ####################################################################################

            info = base_model.model_validate(info_json)
            info_task = info.info.task if isinstance(info, (ModelReportProps, DatasetReportProps)) else info.task
            if tasks is None or info_task in tasks:
                out.append(info)
    return out


@router.post("/upload")
def upload(
        file: dict,
        repository: Literal["model_report", "dataset_report"]
):
    path = Path("~/Desktop/StableAI").expanduser()
    if repository == "model_report":
        try:
            info = ModelReportProps.model_validate(file)
        except ValueError as e:
            print(e)
            raise HTTPException(status_code=400, detail=str(e))

    elif repository == "dataset_report":
        try:
            info = DatasetReportProps.dataset_validate(file)
        except ValueError as e:
            print(e)
            raise HTTPException(status_code=400, detail=str(e))
    else:
        raise HTTPException(status_code=400, detail="Repository not supported.")

    return {}
