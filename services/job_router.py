import json
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Body, HTTPException, Query, Request

from benchmarking import create_benchmark_id, run_benchmark
from models import BenchmarkExecutionConfig, DatasetInfo, ModelInfo, ServerConfig, RegisteredObject, \
    BenchmarkOptionConfig, JobResult

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

    # run_benchmark consumes serializable mappings, not the API metadata
    # model returned by /info/attacks and /info/metrics.
    attacks: list[RegisteredObject] = [
        attack for attack in body.attacks
        if attack.id not in config.excluded_attacks
    ]
    metrics: list[RegisteredObject] = body.metrics
    # The service owns the benchmark repository.  Using its configured path
    # also means getJobs can recover statuses from disk after a restart.
    options: BenchmarkOptionConfig = body.options

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
        attacks_id: list[str] | None = Query(None)
) -> list[JobResult]:
    """
    Get the status of all benchmark attacks, optionally for one benchmark.
    """
    if benchmark_id is None:
        raise HTTPException(status_code=422, detail="benchmark_id is required")

    # Some clients JSON-encode this query parameter, producing %22id%22.
    # Accept that representation as well as the regular unquoted value.
    normalized_benchmark_id = benchmark_id.strip().strip("\"'").strip()
    if not normalized_benchmark_id:
        raise HTTPException(status_code=422, detail="benchmark_id cannot be empty")

    config: ServerConfig = request.app.state.config
    repository = Path(config.path_model_report_repo).expanduser().resolve()
    output_folder = (repository / normalized_benchmark_id).resolve()
    if output_folder.parent != repository:
        raise HTTPException(status_code=400, detail="Invalid benchmark_id")

    # FastAPI uses repeated query parameters for lists.  The web client sends
    # one comma-separated value, so support both forms:
    #   ?attacks_id=a&attacks_id=b
    #   ?attacks_id=a,b
    normalized_attacks = list(dict.fromkeys(
        attack_id.strip()
        for value in (attacks_id or [])
        for attack_id in value.split(",")
        if attack_id.strip()
    ))
    if not normalized_attacks and output_folder.is_dir():
        normalized_attacks = sorted(
            path.name
            for path in output_folder.iterdir()
            if path.is_dir() and (path / "results.json").is_file()
        )

    jobs: list[JobResult] = []

    for atk in normalized_attacks:
        attack_folder = (output_folder / atk).resolve()
        if attack_folder.parent != output_folder:
            raise HTTPException(status_code=400, detail=f"Invalid attack id: {atk}")
        json_file = attack_folder / "results.json"
        job = JobResult(
            id=atk,
        )
        if json_file.exists():
            with open(json_file, encoding="utf-8") as file:
                job = JobResult.model_validate(json.load(file))

        jobs.append(job)

    return jobs
