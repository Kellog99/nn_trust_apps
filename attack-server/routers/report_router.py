import json
import os
from pathlib import Path

from fastapi import APIRouter, Query, Body
from nn_trust.core import Task

from lib.model import ModelReportProps, ReportInfoProps, ReportMetricsProps, ReportAttacksProps, \
    UploadReportModel, BenchmarkModelProps

router = APIRouter(prefix="/report", tags=["datasets and models"])


@router.get("/repository", response_model=list[ModelReportProps])
def get_reports(
        repo_path: str = Query(
            default=...,
            description="Path to the repository"),
        tasks: str | list[str] | None = Query(
            default="Classification",
            description="It represents the task to filter the reports with."
        ),
        datasets: str | list[str] | None = Query(
            default=None,
            description="It represents the dataset to filter the reports with."
        )
) -> list[ModelReportProps]:
    """
    This function handle the fetching of all the reports from a certain path
    It is also possible to filter the list of reports by:
        * task
        * dataset
    """
    out = []
    if os.path.exists(repo_path):
        if datasets and (not isinstance(datasets, list)):
            datasets = [datasets]
        if tasks and (not isinstance(tasks, list)):
            tasks = [tasks]
        for dirpath, dirnames, filenames in os.walk(repo_path):
            if "report.json" in filenames:
                report_path = os.path.join(dirpath, "report.json")
                try:
                    with open(report_path, "r", encoding="utf-8") as f:
                        report = json.load(f)

                    model_report = ModelReportProps(
                        info=ReportInfoProps(**report['info']),
                        metrics=ReportMetricsProps(**report['metrics']),
                        attacks={atk: ReportAttacksProps(**atk_info) for atk, atk_info in
                                 report['attacks'].items()}
                    )
                    # Filtering the reports that satisfies a specific task and dataset
                    # If the task or the dataset is None then there is no filtering
                    # if ((tasks is None or model_report.info.task in tasks)
                    #         and (datasets is None or model_report.info.dataset in datasets)):
                    out.append(model_report)

                except (json.JSONDecodeError, OSError) as e:
                    print(f"⚠️ Could not read {report_path}: {e}")
    return out


@router.get("/listBenchmarking", response_model=ModelReportProps)
def get_list_benchmarks(
        repo_path: str = Query(
            default=...,
            description="Path to the repository"),
        tasks: str | list[str] | None = Query(
            default=Task.Classification.name,
            description="It represents the task to filter the reports with."
        ),
        datasets: str | list[str] | None = Query(
            default=None,
            description="It represents the dataset to filter the reports with."
        )
) -> list:
    """
    This function handle the creation of the list responsible for the ranking list.
    It is also possible to filter the list of reports by:
        * task
        * dataset
    """
    list_reports = get_reports(
        repo_path=repo_path,
        tasks=tasks,
        datasets=datasets
    )
    return [
        BenchmarkModelProps(
            name=report.info.name,
            param=report.info.parameters,
            task=report.info.task,
            benchmark_id=report.info.id,
            metrics=report.metrics.model_dump()
        ) for report in list_reports
    ]


@router.post("/upload/model", response_model=ModelReportProps)
def upload_report(
        report: UploadReportModel = Body(...),
        report_path: str = Query(
            default=...,
            description="Path to the repository's folder"
        )
) -> ModelReportProps:
    """
    This function handles the uploading of the report.
    At this moment, it handles only the Models' report repository.
    """
    print(report)
    report = ModelReportProps(
        info=ReportInfoProps(**report.info),
        metrics=ReportMetricsProps(**report.metrics),
        attacks={atk: ReportAttacksProps(**atk_value) for atk, atk_value in report.attacks.items()}
    )
    new_report_path = Path(report_path) / report.info.id
    # create a folder for the new report
    new_report_path.mkdir(parents=True, exist_ok=True)
    # Create the file path (not just folder)
    file_path = new_report_path / "report.json"  # or whatever filename you want
    # Convert Pydantic model to dict before saving
    with open(file_path, "w") as f:
        json.dump(report.model_dump(), f, indent=2)  # or report.dict() for older Pydantic

    return report
