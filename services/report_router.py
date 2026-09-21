"""
This router handles:
 * the reports' PDF creation
 * the upload of a report
 * the creation of the benchmark
"""
import json
from pathlib import Path
from typing import cast

from fastapi import APIRouter, Query, Body, Depends
from fastapi.responses import StreamingResponse

from models import config_field
from models.model import BenchmarkModelProps
from models.reports import ModelReportProps
from report import AdversarialReportGenerator
from services.repository_router import get_info

router = APIRouter(prefix="/report", tags=["datasets and model"])


@router.get("/benchmarks", response_model=list[BenchmarkModelProps])
def get_benchmarks(
        id: str = Query(
            default=None,
            description="It represents the model's id that is tested. Hence It has to be excluded from the list."
        ),
        tasks: str | list[str] | None = Query(
            default=None,
            description="It represents the task to filter the reports with."
        ),
        datasets: str | list[str] | None = Query(
            default=None,
            description="It represents the dataset to filter the reports with."
        ),
        repo_path: str | Path = Depends(config_field(attr_name="path_model_report_repo")),
) -> list[BenchmarkModelProps]:
    """
    This function handle the creation of the list responsible for the ranking list.
    It is also possible to filter the list of reports by:
        * task
        * dataset
    """
    if isinstance(repo_path, str):
        repo_path: Path = Path(repo_path).expanduser()

    if isinstance(tasks, str):
        tasks = [tasks]
    if isinstance(datasets, str):
        datasets = [datasets]

    list_reports = cast(list[ModelReportProps], get_info(
        tasks=tasks,
        repo_path=repo_path,
        model_type="report_model"
    ))

    out: list[BenchmarkModelProps] = []
    for report in list_reports:
        # There are three level of filtering
        # 1) the id: the id of the selected model must not be inside the benchmark list
        # 2) the task: the model inside the benchmark must satisfy the filter for the task
        # 3) the dataset: the model inside the benchmark must satisfy the filter for the dataset
        report_dataset = getattr(report.info, "dataset", None)
        if ((id is None or report.info.id != id)
                and (tasks is None or report.info.task in tasks)
                and (datasets is None or report_dataset is None or report_dataset in datasets)):
            out.append(
                BenchmarkModelProps(
                    name=report.info.name,
                    param=report.info.parameters,
                    task=report.info.task,
                    benchmark_id=report.info.id,
                    # Rankings require scalar numbers. Registered metrics may
                    # also return vectors, matrices or structured payloads;
                    # preserve those in the full report only.
                    metrics={
                        key: value
                        for key, value in report.metrics.model_dump(
                            exclude_none=True,
                        ).items()
                        if isinstance(value, (int, float))
                        and not isinstance(value, bool)
                    }
                )
            )
    print("list of benchmarks ", out)
    return out


@router.post("/generate_pdf")
def generate_pdf_report(
        data: dict = Body(...)
):
    """
    This function takes all the necessary information for generating the model's report.
    """

    data: ModelReportProps = ModelReportProps.model_validate(data.get("report"))

    file_name: str = data.info.name or data.info.id or "model_adversarial_report.pdf"
    file_name = file_name.replace(" ", "_").lower()
    # Adding the proper extension to the file
    if not file_name.endswith(".pdf"):
        file_name = file_name + ".pdf"

    output_path: Path = Path(getattr(data, "output_path", "./out")).expanduser() / file_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = AdversarialReportGenerator()
    report.generate(
        data=data,
        output_path=output_path,
        header_logo_path=None,
    )

    pdf = report.pdf_to_bytesio(pdf_path=output_path)
    return StreamingResponse(
        pdf,
        media_type="application/pdf",
        headers={
            "Content-Disposition": 'inline; filename="model_report.pdf"'
        },
    )
