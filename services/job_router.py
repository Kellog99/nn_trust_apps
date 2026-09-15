import json
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Body, HTTPException, Query, Request

from benchmarking import create_benchmark_id, run_benchmark
from models import BenchmarkExecutionConfig, DatasetInfo, ModelInfo, ServerConfig, RegisteredObject, \
    BenchmarkOptionConfig, JobResult, ModelReportProps

router = APIRouter(prefix="/job", tags=["jobs management", "jobs utils"])


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
    dataset: DatasetInfo = body.dataset
    model: ModelInfo = body.model
    benchmark_id: str = create_benchmark_id()

    attacks: list[RegisteredObject] = [
        attack for attack in body.attacks
        if attack.id not in config.excluded_attacks
    ]

    metrics: list[RegisteredObject] = body.metrics
    # Benchmark artifacts belong to the server repository used by get_jobs.
    options: BenchmarkOptionConfig = body.options.model_copy(
        update={"output_path": config.path_model_report_repo}
    )

    attack_ids = {attack.id for attack in attacks}
    # run_benchmark always executes the identity baseline, even when it was
    # not explicitly included in the request.
    attack_ids.add("identitybaseline")
    benchmark_folder = (
            Path(options.output_path).expanduser().resolve()
            / benchmark_id
            / model.id
            / dataset.id
    )
    for attack_id in attack_ids:
        (benchmark_folder / attack_id).mkdir(parents=True, exist_ok=True)

    background_tasks.add_task(
        run_benchmark,
        models=[model],
        datasets=[dataset],
        attacks=attacks,
        metrics=metrics,
        options=options,
        benchmark_id=benchmark_id,
    )

    return benchmark_id


# --- Progress --- #
@router.get("/getJobs")
def get_jobs(
        request: Request,
        benchmark_id: str | None = Query(None),
        model_id: str | None = Query(None),
        dataset_id: str | None = Query(None),
        attacks_id: list[str] | None = Query(None)
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

    if dataset_id is None:
        raise HTTPException(status_code=422, detail="dataset_id is required")
    if model_id is None:
        raise HTTPException(status_code=422, detail="model_id is required")

    output_folder: Path = output_folder / benchmark_id / model_id / dataset_id

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

    for atk in attacks_id:
        json_file: Path = output_folder / atk / "job_results.json"
        print(json_file)
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
