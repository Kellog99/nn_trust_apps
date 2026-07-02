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

from fastapi import APIRouter, Query, Depends, Request, Body
from pydantic import BaseModel, ValidationError

from models import config_field, ModelReportProps, DatasetReportProps, ModelInfo, DatasetInfo

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
):
    """
    Get all models or dataset that are saved in the `repo_path` that satisfy the filter given by the `tasks`.
    It follows the general scaffolding that was defining in the beginning.

    Args:
        tasks (str | list[str] | None )
        file_checker
        repo_path
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
    print(f"Exploring {repo_path}")
    for root, dirs, files in os.walk(repo_path):
        if file_checker in files:
            file_info_path = os.path.join(root, file_checker)
            with open(file_info_path, "r") as f:
                info_json = json.load(f)
            # Dynamically saving all the absolute path for the file
            info_json["repository"] = root

            ############################ Instance of the base model ############################

            def get_model(data: dict):
                for model in [DatasetInfo, ModelInfo, ModelReportProps, DatasetReportProps]:
                    try:
                        return model.model_validate(data)
                    except ValidationError:
                        continue
                raise ValueError("No model among the possible models is valid. Please retry.")

            ####################################################################################

            info = get_model(info_json)
            info_task = info.info.task if isinstance(info, (ModelReportProps, DatasetReportProps)) else info.task
            if tasks is None or info_task in tasks:
                out.append(info)
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
