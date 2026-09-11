import logging

from fastapi import APIRouter, Response, Body, Query, Request

from benchmarking import run_benchmark, executor
from models import BenchmarkExecutionConfig, DatasetInfo, ModelInfo, ServerConfig

router = APIRouter(prefix="/job", tags=["jobs management", "jobs utils"])


@router.post("/start_benchmark")
def start_benchmark_job(
        request: Request,
        body: BenchmarkExecutionConfig = Body(...)
) -> list[dict]:
    """
    Start a new TITANN benchmark job.
    """

    config: ServerConfig = request.app.state.config
    dataset: DatasetInfo = body.dataset
    model: ModelInfo = body.model

    # run_benchmark consumes serializable mappings, not the API metadata
    # model returned by /info/attacks and /info/metrics.
    attacks = [
        attack.model_dump(exclude_none=True)
        for attack in body.attacks if
        attack.id not in config.excluded_attacks
    ]
    metrics = [metric.model_dump(exclude_none=True) for metric in body.metrics]
    options = body.options

    result = run_benchmark(
        models=[model],
        datasets=[dataset],
        attacks=attacks,
        metrics=metrics,
        options=options
    )

    return [report.model_dump(mode="json") for report in result]


# --- Progress --- #
@router.get("/getJobs")
def get_jobs(id: str = Query(None)):
    """
    Get the status of all benchmark attacks, optionally for one benchmark.
    """
    try:
        tasks = executor.tracker.list_tasks()
        if id:
            id = id.replace(" ", "")
            return [
                {
                    "id": task["attack_id"],
                    "name": task["name"],
                    "status": task["status"],
                    "progress": task["progress"],
                    "error": task["error"],
                }
                for task in tasks.values()
                if task["benchmark_id"] == id
            ]
        return tasks

    except Exception as e:
        logging.error(f"Unexpected error during get jobs: {str(e)}")
        return Response(
            status_code=500,
            content=f"Unexpected error during get jobs"
        )
