import json
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Body, HTTPException, Query, Request

from benchmarking import create_benchmark_id, run_benchmark
from models import BenchmarkExecutionConfig, JobResult, ModelReportProps, ServerConfig

router = APIRouter(prefix="/job", tags=["jobs management", "jobs utils"])


def _run_benchmark_background(
        benchmark: BenchmarkExecutionConfig,
        benchmark_id: str,
        benchmark_folder: Path,
) -> None:
    """
    Run a benchmark without leaking background-task failures into ASGI.
    """
    try:
        run_benchmark(
            models=[benchmark.model],
            datasets=[benchmark.dataset],
            attacks=benchmark.attacks,
            metrics=benchmark.metrics,
            options=benchmark.options,
            benchmark_id=benchmark_id,
        )
    except Exception as exc:
        # In case the background application fails due to various reason,
        # the output is saved as an error
        error = f"{type(exc).__name__}: {exc}"

        for attack_id in {attack.id for attack in benchmark.attacks} | {"identitybaseline"}:
            result_file = benchmark_folder / attack_id / "job_results.json"
            try:
                job = JobResult(id=attack_id)
                if result_file.exists():
                    job = JobResult.model_validate_json(result_file.read_text(encoding="utf-8"))
                if job.status not in {"finished", "error"}:
                    job.status, job.error = "error", error
                    job.save(result_file)
            except (OSError, ValueError):
                print("Could not persist failed job '%s'", attack_id)


@router.post("/start_benchmark")
async def start_benchmark_job(
        request: Request,
        background_tasks: BackgroundTasks,
        body: BenchmarkExecutionConfig = Body(...)
) -> str:
    """
    Start a new benchmark.
    """

    config: ServerConfig = request.app.state.config
    benchmark_id: str = create_benchmark_id()
    benchmark = body.model_copy(update={
        "attacks": [
            attack
            for attack in body.attacks
            if attack.id not in config.excluded_attacks
        ],
        "options": body.options.model_copy(
            update={"output_path": config.path_model_report_repo}
        ),
    })
    benchmark_folder: Path = (
            Path(benchmark.options.output_path).expanduser().resolve()
            / benchmark_id
            / benchmark.model.id
            / benchmark.dataset.id
    )
    # The identity baseline runs even when omitted from the request.
    for attack_id in {attack.id for attack in benchmark.attacks} | {"identitybaseline"}:
        (benchmark_folder / attack_id).mkdir(parents=True, exist_ok=True)

    print(benchmark.dataset)
    background_tasks.add_task(
        _run_benchmark_background,
        benchmark=benchmark,
        benchmark_folder=benchmark_folder,
        benchmark_id=benchmark_id,
    )

    return benchmark_id


# --- Progress --- #
@router.get("/getJobs")
def get_jobs(
        request: Request,
        benchmark_id: Annotated[str | None, Query()] = None,
        model_id: Annotated[str | None, Query()] = None,
        dataset_id: Annotated[str | None, Query()] = None,
        attacks_id: Annotated[list[str] | None, Query()] = None
) -> list[JobResult]:
    """
    Get the status of all benchmark attacks, optionally for one benchmark.
    """
    if benchmark_id is None:
        raise HTTPException(status_code=422, detail="benchmark_id is required")

    # Some clients JSON-encode this query parameter, producing %22id%22.
    # Accept that representation as well as the regular unquoted value.
    benchmark_id = benchmark_id.strip().strip("\"'").strip()
    if not benchmark_id:
        raise HTTPException(status_code=422, detail="benchmark_id cannot be empty")

    config: ServerConfig = request.app.state.config
    output_folder: Path = Path(config.path_model_report_repo).expanduser().resolve()

    if (model_id is None) != (dataset_id is None):
        raise HTTPException(status_code=422, detail="model_id and dataset_id must be provided together")
    repository = output_folder
    output_folder = output_folder / benchmark_id
    if model_id is not None:
        output_folder = output_folder / model_id / dataset_id
    output_folder = output_folder.resolve()
    if repository not in output_folder.parents:
        raise HTTPException(status_code=400, detail="Invalid benchmark path")

    # FastAPI builds a list from repeated query parameters, but some clients
    # send all attack IDs in a single comma-separated value. Support both:
    #   ?attacks_id=a&attacks_id=b
    #   ?attacks_id=a,b
    attacks_id = [
        attack_id.strip().strip("\"'").strip()
        for value in (attacks_id or [])
        for attack_id in value.split(",")
        if attack_id.strip().strip("\"'").strip()
    ]

    jobs: list[JobResult] = []

    if not attacks_id:
        return [
            JobResult.model_validate_json(path.read_text(encoding="utf-8"))
            for path in sorted(output_folder.rglob("job_results.json"))
        ]

    for atk in attacks_id:
        json_file: Path = output_folder / atk / "job_results.json"
        job = JobResult(id=atk)
        if json_file.exists():
            with open(json_file, encoding="utf-8") as file:
                job = JobResult.model_validate(json.load(file))
        jobs.append(job)
    return jobs


@router.get("/getReport")
def getReport(
        request: Request,
        benchmark_id: str = Query(...),
        model_id: str = Query(...),
        dataset_id: str = Query(...),
) -> ModelReportProps:
    """
    Load a benchmark report for a model and dataset.
    """
    config: ServerConfig = request.app.state.config
    repository = Path(config.path_model_report_repo).expanduser().resolve()

    ids = {
        "benchmark_id": benchmark_id,
        "model_id": model_id,
        "dataset_id": dataset_id,
    }
    normalized_ids = {
        name: value.strip().strip("\"'").strip()
        for name, value in ids.items()
    }
    for name, value in normalized_ids.items():
        if not value:
            raise HTTPException(status_code=422, detail=f"{name} cannot be empty")

    report_file = repository.joinpath(*normalized_ids.values(), "report.json").resolve()
    if repository not in report_file.parents:
        raise HTTPException(status_code=400, detail="Invalid report path")
    if not report_file.is_file():
        raise HTTPException(status_code=404, detail="Report not found")

    with report_file.open(encoding="utf-8") as file:
        return ModelReportProps.model_validate(json.load(file))
