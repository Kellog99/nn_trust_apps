import logging
from pathlib import Path
from fastapi import APIRouter, Response, Body, Query, Request

from benchmarking import create_benchmark_id, run_benchmark, executor
from models import BenchmarkExecutionConfig, DatasetInfo, ModelInfo, ServerConfig, TaskStatus, RegisteredObject, \
    BenchmarkOptionConfig

router = APIRouter(prefix="/job", tags=["jobs management", "jobs utils"])


@router.post("/start_benchmark")
def start_benchmark_job(
        request: Request,
        body: BenchmarkExecutionConfig = Body(...)
) -> str:
    """
    Start a new benchmark.
    """

    config: ServerConfig = request.app.state.config
    dataset: DatasetInfo = body.dataset
    model: ModelInfo = body.model

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

    benchmark_id = create_benchmark_id()
    run_benchmark(
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
def get_jobs(id: str = Query(None), request: Request = None):
    """
    Get the status of all benchmark attacks, optionally for one benchmark.
    """
    try:
        tasks: dict[str, TaskStatus] = executor.tracker.list_tasks()
        if id:
            id = id.replace(" ", "")
            matching_tasks = {
                task_id: task
                for task_id, task in tasks.items()
                if task["benchmark_id"] == id
            }
            if not matching_tasks and request is not None:
                config: ServerConfig = request.app.state.config
                benchmark_path = Path(config.path_model_report_repo) / id
                matching_tasks = executor.tracker.list_benchmark(benchmark_path)
            return [
                {
                    "id": task["attack_id"],
                    "name": task["name"],
                    "status": task["status"],
                    "progress": task["progress"],
                    "error": task["error"],
                }
                for task in matching_tasks.values()
            ]

        if request is not None:
            config: ServerConfig = request.app.state.config
            tasks.update(executor.tracker.list_repository(config.path_model_report_repo))
        return tasks

    except Exception as e:
        logging.error(f"Unexpected error during get jobs: {str(e)}")
        return Response(
            status_code=500,
            content=f"Unexpected error during get jobs"
        )
