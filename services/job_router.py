import base64
import json
import logging
import os

import ray
import requests
from fastapi import APIRouter, Response, Body, Query

from benchmarking import run_benchmark, executor
from models import BenchmarkExecutionConfig, DatasetInfo, ModelInfo, RegisteredObject

router = APIRouter(prefix="/job", tags=["jobs management", "jobs utils"])


@router.post("/start_benchmark")
async def start_benchmark_job(body: BenchmarkExecutionConfig = Body(...)) -> dict:
    """
    Start a new TITANN benchmark job.
    """

    dataset: DatasetInfo = body.dataset
    model: ModelInfo = body.model
    attacks: list[RegisteredObject] = body.attacks
    metrics: list[RegisteredObject] = body.metrics
    options = body.options

    result = run_benchmark(
        models=[model],
        datasets=[dataset],
        attacks=attacks,
        metrics=metrics,
        options=options
    )

    result["output_path"] = str(result["output_path"])
    return result


# --- Progress --- #
@router.get("/getJobs")
def get_jobs(id: str = Query(None)):
    """
    Get all running benchmark jobs in the TITANN backend.
    """
    try:
        tasks = ray.get(executor.tracker.list_tasks.remote())
        if id:
            id = id.replace(" ", "")
            output = []
            if tasks:
                for k, v in tasks.items():
                    output_dict = {}
                    if v["benchmark_id"] == id:
                        atk_id = k.split(f"_{v['benchmark_id']}")[0]
                        output_dict["id"] = atk_id
                        output_dict["name"] = router.state.attacks[
                            atk_id].name if atk_id != "reference" else "Reference (Identity Attack)"
                        output_dict["status"] = v["status"]
                        output_dict["progress"] = v["progress"]
                        if output_dict:
                            output.append(output_dict)
            return output
        else:
            return tasks

    except Exception as e:
        logging.error(f"Unexpected error during get jobs: {str(e)}")
        return Response(
            status_code=500,
            content=f"Unexpected error during get jobs"
        )


# --- Results --- #
@router.get("/report/getResult")
def get_jobs_results(
        id: str = Query(
            default=None,
            description="report id"
        ),
        dataset: str = Query(
            default=None,
            description="Dataset of the results"
        ),
        model: str = Query(
            default=None,
            description="Model that has to filter for the results"
        ),
        pdf_report: bool = Query(
            default=False,
            description="This flag tells whether a pdf report has to be done."
        )):
    """
    Get a TITANN benchmark report job result.
    """
    try:

        model_dir, task_dir = "s", "aa"  # find_model_and_task_dir(os.environ.get("BENCHMARK_OUTPUT_DIR"), dataset, model, id)
        benchmark_id = task_dir.split(os.sep)[-1]
        tasks = {k: v for k, v in ray.get(executor.tracker.list_tasks.remote()).items() if
                 v["benchmark_id"] == benchmark_id}

        if not tasks:
            logging.error(f"Benchmark {benchmark_id} not found")
            return Response(
                status_code=404,
                content=f"Benchmark {benchmark_id} not found"
            )
        benchmarking.postprocess_benchmark_run_results(task_dir)
        with open(os.path.join(model_dir, 'info.json'), "r", encoding="utf-8") as f:
            info = json.load(f)
        info["dataset"] = str(model_dir).split(os.sep)[-2]
        info["id"] = benchmark_id
        info["task"] = "Classification"

        # ----# thumbnail
        prototype = json.loads(requests.get(
            f"http://{os.getenv('DQ_HOST')}:{os.getenv('DQ_PORT')}/getDataset?dataset=animals").text)[
            "prototype"]["datas"][0]
        # ----#

        with open(os.path.join(model_dir, 'aggregate_statistics.json'), "r", encoding="utf-8") as f:
            aggregate = json.load(f)
            aggregate["params"] = info["parameters"]

        statistics = {}
        for entry in os.listdir(model_dir):
            entry_path = os.path.join(model_dir, entry)
            if os.path.isdir(entry_path):
                stat_file = os.path.join(entry_path, "statistics.json")
                if os.path.exists(stat_file) and os.path.isfile(stat_file):
                    try:
                        with open(stat_file, "r", encoding="utf-8") as sf:
                            sf_data = json.load(sf)
                            sf_data["name"] = router.state.attacks[
                                entry.lower()].name if entry in router.state.attacks else entry
                            statistics[entry.upper()] = sf_data
                    except Exception as e:
                        logging.warning(f"Could not load statistics.json in '{entry_path}': {e}")

        report_data = {
            "info": info,
            "metrics": aggregate,
            "attacks": statistics
        }
        if prototype:
            report_data["prototype"] = prototype

        with open(os.path.join(task_dir, "report.json"), "w", encoding="utf-8") as f:
            json.dump(report_data, f)

        if pdf_report:
            generator = benchmarking.AdversarialReportGenerator(logo_path='./resources/logo_leonardo.png')
            report_file = './resources/adversarial_report.pdf'
            generator.generate(report_data, report_file)
            with open(report_file, 'rb') as pdf_file:
                pdf_bytes = pdf_file.read()
                return base64.b64encode(pdf_bytes).decode('utf-8')
        else:
            return report_data


    except Exception as e:
        logging.error(f"Unexpected error during get result: {str(e)}")
        return Response(
            status_code=500,
            content=f"Unexpected error during get result"
        )
