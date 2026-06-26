import base64
import importlib
import json
import logging
import os

import ray
import requests
from fastapi import APIRouter, Response
from fastapi import Body, Query
from fastapi.responses import JSONResponse

from lib.disk_reader import find_model_and_task_dir
from lib.model import ExecutionConfig
from utils.utils import find_image

benchmarking = importlib.import_module("benchmarking")

router = APIRouter(prefix="/job", tags=["jobs management", "jobs utils"])

if not hasattr(router, "state"):
    class _RouterState:
        pass


    router.state = _RouterState()

if not hasattr(router.state, "attacks"):
    router.state.attacks = benchmarking.get_attacks_info()

# Setting up number of actors and executor
num_actors = int(os.environ.get("RAY_NUM_ACTORS", 1))
num_gpu_per_actors = float(os.environ.get("FRACTION_FOR_GPU_ACTOR", 1))
modules = os.environ.get("RAY_PY_MODULES", None)
if modules:
    ray.init(ignore_reinit_error=True, runtime_env={
        "py_modules": [modules]
    })
executor = benchmarking.benchmark_utils.executor.RayActorPoolExecutor(num_actors=num_actors,
                                                                      num_gpus_per_actor=num_gpu_per_actors)


# ----------------- SERVICES --------------------------

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
                        atk_id = k.split(f"_{v["benchmark_id"]}")[0]
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

        model_dir, task_dir = find_model_and_task_dir(os.environ.get("BENCHMARK_OUTPUT_DIR"), dataset, model, id)
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
        try:
            prototype = json.loads(requests.get(
                f"http://{os.getenv('DQ_HOST')}:{os.getenv('DQ_PORT')}/getDataset?dataset=animals").text)[
                "prototype"]["datas"][0]
        except Exception as e:
            prototype = find_image(os.path.join(os.environ.get("DATASETS_REPO"), str(model_dir).split(os.sep)[-2]))
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


# --- Start --- #
@router.post("/start", response_model=str)
async def start_benchmark_job(body: ExecutionConfig = Body(...)) -> str | Response:
    """
    Start a new TITANN benchmark job.
    """
    try:
        dataset_name = body.dataset
        model_name = body.model

        # Branching logic for NLP tasks
        if body.task_type == "nlp":
            # NLP-specific path
            file_path = os.path.join(os.environ.get('DATASETS_REPO'), dataset_name, f"{dataset_name}.json")
            # For NLP, source_path is the directory of the JSON dataset
            source_path = os.path.join(os.environ.get('DATASETS_REPO'), dataset_name)
        else:
            # Original Image-based path
            file_path = os.path.join(os.environ.get('DATASETS_REPO'), dataset_name, f"{dataset_name}.json")
            source_path = os.path.join(dataset_name, "data")

        if os.path.exists(file_path) and os.path.isfile(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                config_dataset = json.load(f)  # load into dict
                config_dataset["name"] = dataset_name
            logging.info("Loaded JSON metadata")
        else:
            logging.error(f"Can't find dataset metadata in the repository for dataset:{dataset_name}")
            return Response(
                status_code=404,
                content=f"Can't find dataset metadata in the repository for dataset:{dataset_name}"
            )

        benchmark_config = {}
        benchmark_config['options'] = {
            "load_results": os.environ.get("BENCHMARK_LOAD_RESULTS", "False").lower() == "true",
            "overwrite": os.environ.get("BENCHMARK_OVERWRITE", "True").lower() == "true",
            "num_images_to_save": int(os.environ.get("BENCHMARK_NUM_IMAGES_TO_SAVE", "-1")),
            "save_perturbation": os.environ.get("BENCHMARK_SAVE_PERTURBATION", "False").lower() == "true",
            "gpu": os.environ.get("BENCHMARK_GPU", "True").lower() == "true",
            "output_path": os.environ.get("BENCHMARK_OUTPUT_DIR", "./benchmark_out"),
            "output_format": os.environ.get("BENCHMARK_OUTPUT_FORMAT", "report")
        }

        model_file_path = os.path.join(os.environ.get('MODEL_REPO'), model_name)
        config_models = {
            "model_path": model_file_path
        }

        config_dataset["source_path"] = source_path

        benchmark_config['datasets'] = [config_dataset]
        benchmark_config['models'] = [config_models]
        benchmark_config['attacks'] = [
            {
                "name": attack["id"],
                **{param["id"]: param["default"] for param in attack.get("parameters", [])}
            }
            for attack in body.attacks
        ]
        benchmark_config['evaluation'] = {}
        benchmark_config["evaluation"]["statistics"] = [
            {
                "name": metric["id"],
                **{param["id"]: param["default"] for param in metric.get("parameters", [])}
            }
            for metric in body.metrics
        ]
        benchmark_config["task_type"] = body.task_type
        task_id = benchmarking.benchmark(benchmark_config, executor)
        return JSONResponse(status_code=200, content=task_id)

    except Exception as e:
        logging.error(f"Unexpected error during job start: {str(e)}")
        raise e


