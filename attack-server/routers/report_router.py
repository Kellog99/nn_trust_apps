import json
import os
from pathlib import Path

from fastapi import APIRouter, Query, Body

from lib.model import ModelReportProps, ReportInfoProps, ReportMetricsProps, ReportAttacksProps, \
    UploadReportModel

router = APIRouter(prefix="/report", tags=["datasets and models"])


@router.get("/repository", response_model=list[ModelReportProps])
def get_reports(
        repo_path: str = Query(
            default=...,
            description="Path to the repository"),
        isModel: bool = Query(
            default=...,
            description="Whether to parse as model or dataset report")
) -> list[ModelReportProps]:
    """
    This function has to take all the report from a certain path
    """
    print(repo_path)
    if os.path.exists(repo_path):
        out = []
        for dirpath, dirnames, filenames in os.walk(repo_path):
            if "report.json" in filenames:
                report_path = os.path.join(dirpath, "report.json")
                try:
                    with open(report_path, "r", encoding="utf-8") as f:
                        report = json.load(f)
                    out.append(
                        ModelReportProps(
                            info=ReportInfoProps(**report['info']),
                            metrics=ReportMetricsProps(**report['metrics']),
                            attacks={atk: ReportAttacksProps(**atk_info) for atk, atk_info in
                                     report['attacks'].items()}
                        )
                    )
                except (json.JSONDecodeError, OSError) as e:
                    print(f"⚠️ Could not read {report_path}: {e}")
        print("THe output is ", out)
        return out
    return []


@router.post("/upload/model", response_model=ModelReportProps)
def load_reports(
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
