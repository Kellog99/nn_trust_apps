import base64
import io
import json
import logging
import os
import time
from typing import Union

import ray
import torch
from PIL import Image
from fastapi import APIRouter, Response, Request, HTTPException
from fastapi import Body, Query
from fastapi.responses import JSONResponse
from nn_trust.attack.attack_factory import EvasionAttackFactory as EAF
from nn_trust.core import Task, ModelAdapter
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchvision import transforms

from lib.disk_reader import find_model_and_task_dir
from lib.model import Error, ExecutionConfig, SingleAttackOutput, SingleAttackProps
from routers.dataset_router import get_datasets
from routers.model_router import get_models
from routers.utils import find_image

benchmarking = ""  # importlib.import_module("benchmarking")

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
            content=Error(code=500, message=f"Unexpected error during get jobs").model_dump_json())


# --- Results --- #
@router.get("/report/getResult")
def get_jobs_results(id: str = Query(None),
                     dataset: str = Query(None),
                     model: str = Query(None),
                     pdf_report: bool = Query(None)):
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
                content=Error(code=404, message=f"Benchmark {benchmark_id} not found").model_dump_json())

        if False:
            with open(os.path.join(task_dir, "report.json"), 'r', encoding='utf-8') as f:
                report_data = json.load(f)
            prototype = None
        else:
            benchmarking.postprocess_benchmark_run_results(task_dir)
            with open(os.path.join(model_dir, 'info.json'), "r", encoding="utf-8") as f:
                info = json.load(f)

            # ----# thumbnail
            try:
                import requests
                prototype = json.loads(requests.get(
                    f"http://{os.getenv('DQ_HOST')}:{os.getenv('DQ_PORT')}/getDataset?dataset=animals").text)[
                    "prototype"]["datas"][0]
            except Exception as e:
                prototype = find_image(os.path.join(os.environ.get("DATASETS_REPO"), str(model_dir).split(os.sep)[-2]))
            # ----#

            with open(os.path.join(model_dir, 'aggregate_statistics.json'), "r", encoding="utf-8") as f:
                aggregate = json.load(f)
                aggregate["params"] = info["parameters"]
                results = benchmarking.collect_dataset_aggregates_with_info(
                    base_dir=os.environ.get("BENCHMARK_OUTPUT_DIR"),
                    dataset=str(model_dir).split(os.sep)[-2],
                    keep_latest_only=False,
                )

                out = benchmarking.transform_to_benchmark(results, task="classification")
                out = benchmarking.enrich_with_ranks(out)
                out = benchmarking.extract_rank_metrics(out, str(model_dir).split(os.sep)[-1])
                num_b = len(results)
                out["total benchmarks"] = num_b
                aggregate = aggregate | out

            statistics = {}
            for entry in os.listdir(model_dir):
                entry_path = os.path.join(model_dir, entry)
                if os.path.isdir(entry_path):
                    stat_file = os.path.join(entry_path, "statistics.json")
                    if os.path.exists(stat_file) and os.path.isfile(stat_file):
                        try:
                            with open(stat_file, "r", encoding="utf-8") as sf:
                                sf_data = json.load(sf)
                                sf_data["name"] = router.state.attacks[entry.lower()].name
                                sf_data["risk"] = 0.5
                                sf_data["num_queries"] = 1
                                sf_data["power"] = 0.5
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
            report_data["tool"] = "nntrust"
            report_data["dataset"] = str(model_dir).split(os.sep)[-2]

            with open(os.path.join(task_dir, "report.json"), "w", encoding="utf-8") as f:
                json.dump(report_data, f)

        if pdf_report and bool(pdf_report) == True:
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
            content=Error(code=500, message=f"Unexpected error during get result").model_dump_json())


@router.get("/benchmark/getResult")
def get_jobs_results(
        dataset: str = Query(...),
        task: str = Query(None),
        id: str = Query(None)
):
    """
    Get a TITANN benchmark job result.
    """
    try:
        tasks = ray.get(executor.tracker.list_tasks.remote())

        if not tasks:
            logging.error(f"No benchmarks found")
            return Response(
                status_code=404,
                content=Error(code=404, message=f"No benchmarks found").model_dump_json())

        results = benchmarking.collect_dataset_aggregates_with_info(
            base_dir=os.environ.get("BENCHMARK_OUTPUT_DIR"),
            dataset=dataset,
            keep_latest_only=False
        )

        out = benchmarking.transform_to_benchmark(results, task="classification")
        if id:
            return [o for o in benchmarking.enrich_with_ranks(out) if o["benchmark_id"] != id]
        else:
            return benchmarking.enrich_with_ranks(out)

    except Exception as e:
        logging.error(f"Unexpected error during get result: {str(e)}")
        return Response(
            status_code=500,
            content=Error(code=500, message=f"Unexpected error during get result").model_dump_json())


# @router.get("/report/getReports")
# def get_reports() -> list[ReportProps]:
#    """
#        Recursively searches for all 'report.json' files under the given root folder
#        and returns a list of their parsed JSON contents (as dictionaries).
#
#        Returns:
#            list[dict]: A list of dictionaries loaded from each report.json file.
#        """
#    reports = []
#    for dirpath, dirnames, filenames in os.walk(os.enviroos.walkn.get("BENCHMARK_OUTPUT_DIR")):
#        if "report.json" in filenames:
#            report_path = os.path.join(dirpath, "report.json")
#            try:
#                with open(report_path, "r", encoding="utf-8") as f:
#                    data = json.load(f)
#                    reports.append(ReportProps(**data))
#            except (json.JSONDecodeError, OSError) as e:
#                print(f"⚠️ Could not read {report_path}: {e}")
#    return reports


################################### POST ###################################
@router.post("/report/upload")
async def upload_report(request: Request):
    try:
        report = await request.json()
        if "info" not in report:
            raise Exception("Info not in the report.")
        if "name" not in report["info"]:
            raise Exception("Model name not in the report.")
        if "dataset" not in report:
            raise Exception("Dataset name not in the report.")

        _, task_dir = find_model_and_task_dir(os.environ.get("BENCHMARK_OUTPUT_DIR"), report["dataset"],
                                              report["info"]["name"], None)
        with open(os.path.join(task_dir, "report.json"), 'w') as f:
            json.dump(report, f, indent=2)
        return Response()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving JSON: {str(e)}")


# --- Start --- #
@router.post("/start", response_model=str)
async def start_benchmark_job(body: ExecutionConfig = Body(...)) -> Union[str, Error]:
    """
    Start a new TITANN benchmark job.
    """
    try:
        dataset_name = body.dataset
        model_name = body.model
        m_response = get_models()
        d_response = get_datasets()
        model_response = m_response

        # if m_response.status_code != 200 or d_response.status_code != 200:
        #    logging.error("Model or Dataset repository is empty. Check repository.")
        #    return Response(
        #        status_code=404,
        #        content=Error(code=404,
        #                      message="Model or Dataset repository is empty. Check repository.").model_dump_json())
        model = [m["name"] for m in model_response if m["name"] == model_name]
        model_type = [m["type"] for m in model_response if m["name"] == model_name]
        dataset = [d["name"] for d in d_response if d["name"] == dataset_name]

        if len(model) > 1 or len(model_type) > 1:
            logging.error(f"The provided model string {model_name} has had more than one match. Check model repository")
            return Response(
                status_code=409,
                content=Error(code=409,
                              message=f"The provided model string {model_name} has had more than one match. Check model repository").model_dump_json())

        if len(model) == 0 or len(model_type) == 0:
            logging.error(f"No model found: {model_name}")
            return Response(
                status_code=404,
                content=Error(code=404,
                              message=f"No model found: {model_name}").model_dump_json())

        if len(dataset) > 1:
            logging.error(
                f"The provided dataset string {dataset_name} has had more than one match. Check dataset repository")
            return Response(
                status_code=409,
                content=Error(code=409,
                              message=f"The provided dataset string {dataset_name} has had more than one match. Check dataset repository").model_dump_json())

        if len(dataset) == 0:
            logging.error(f"No dataset found: {dataset_name}")
            return Response(
                status_code=404,
                content=Error(code=404,
                              message=f"No dataset found: {dataset_name}").model_dump_json())

        file_path = os.path.join(os.environ.get('INTERNAL_MODEL_STORAGE'), f"{model_name}.json")

        if model_type[0] == "saved_model":
            if os.path.exists(file_path) and os.path.isfile(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    config_models = json.load(f)  # load into dict
                    config_models["type"] = "saved_model"
                logging.info("Loaded JSON metadata")
            else:
                logging.error(f"Can't find model metadata in the repository for model:{model_name}")
                return Response(
                    status_code=404,
                    content=Error(code=404,
                                  message=f"Can't find model metadata in the repository for model:{model_name}").model_dump_json())

        if model_type[0] == "hf":
            if os.path.exists(file_path) and os.path.isfile(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    config_models = json.load(f)  # load into dict
                    config_models["type"] = "hf"
                logging.info("Loaded JSON metadata")
            else:
                logging.error(f"Can't find model metadata in the repository for model:{model_name}")
                return Response(
                    status_code=404,
                    content=Error(code=404,
                                  message=f"Can't find model metadata in the repository for model:{model_name}").model_dump_json())

        elif model_type[0] == "timm":
            model_metadata = None
            for m in model_response:
                if m["name"] == model_name:
                    model_metadata = m
            if model_metadata:
                config_models = model_metadata
            else:
                logging.error("An error has occurred for timm model metadata retrieval")
                return Response(
                    status_code=404,
                    content=Error(code=404,
                                  message="An error has occurred for timm model metadata retrieval").model_dump_json())

        file_path = os.path.join(os.environ.get('INTERNAL_DS_STORAGE'), dataset_name, f"{dataset_name}.json")
        if os.path.exists(file_path) and os.path.isfile(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                config_dataset = json.load(f)  # load into dict
                config_dataset["name"] = dataset_name
            logging.info("Loaded JSON metadata")
        else:
            logging.error(f"Can't find dataset metadata in the repository for dataset:{dataset_name}")
            return Response(
                status_code=404,
                content=Error(code=404,
                              message=f"Can't find dataset metadata in the repository for dataset:{dataset_name}").model_dump_json())

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

        if model_type[0] == "saved_model":
            config_models["weights_path"] = os.path.join(os.environ.get('INTERNAL_MODEL_STORAGE'), f"{model_name}.pth")

        if model_type[0] == "hf":
            # TODO:fix
            config_models["name"] = f"timm/{model_name}"
            config_models["weights_path"] = os.path.join(os.environ.get('INTERNAL_MODEL_STORAGE'), f"{model_name}.ckpt")

        config_dataset["source_path"] = os.path.join(dataset_name, "data")

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
        print(benchmark_config['attacks'])
        task_id = benchmarking.benchmark(benchmark_config, executor)
        return JSONResponse(status_code=200, content=task_id)

    except Exception as e:
        logging.error(f"Unexpected error during job start: {str(e)}")
        return Response(
            status_code=500,
            content=Error(code=500, message=f"Unexpected error during job start").model_dump_json())


# --- Single attack --- #
@router.post("/attack")
async def startSingleAttack(body: SingleAttackProps = Body(...)) -> SingleAttackOutput | Error:
    """
    Start a new TITANN attack on single image job.
    """
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ###################### Input ######################
        # Decode base64 image string and convert to torch tensor
        image_bytes = base64.b64decode(body.image)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        x: torch.Tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)

        ############## Extracting the model 2.0 (or as it was before?...) ###########
        # Get the model
        model_name = body.model_name
        model_response = get_models()
        model = [m["name"] for m in model_response if m["name"] == model_name]
        model_type = [m["type"] for m in model_response if m["name"] == model_name]
        file_path = os.path.join(os.environ.get('INTERNAL_MODEL_STORAGE'), f"{model_name}.json")

        if model_type[0] == "saved_model":
            if os.path.exists(file_path) and os.path.isfile(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    config_models = json.load(f)  # load into dict
                    config_models["type"] = "saved_model"
                logging.info("Loaded JSON metadata")
            else:
                logging.error(f"Can't find model metadata in the repository for model:{model_name}")
                return Response(
                    status_code=404,
                    content=Error(code=404,
                                  message=f"Can't find model metadata in the repository for model:{model_name}").model_dump_json())

        elif model_type[0] == "hf":
            if os.path.exists(file_path) and os.path.isfile(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    config_models = json.load(f)  # load into dict
                    config_models["type"] = "hf"
                logging.info("Loaded JSON metadata")
            else:
                logging.error(f"Can't find model metadata in the repository for model:{model_name}")
                return Response(
                    status_code=404,
                    content=Error(code=404,
                                  message=f"Can't find model metadata in the repository for model:{model_name}").model_dump_json())

        elif model_type[0] == "timm":
            model_metadata = None
            for m in model_response:
                if m["name"] == model_name:
                    model_metadata = m
            if model_metadata:
                config_models = model_metadata
            else:
                logging.error("An error has occurred for timm model metadata retrieval")
                return Response(
                    status_code=404,
                    content=Error(code=404,
                                  message="An error has occurred for timm model metadata retrieval").model_dump_json())

        if model_type[0] == "saved_model":
            config_models["weights_path"] = os.path.join(os.environ.get('INTERNAL_MODEL_STORAGE'), f"{model_name}.pth")

        if model_type[0] == "hf":
            # TODO:fix
            config_models["name"] = f"timm/{model_name}"
            config_models["weights_path"] = os.path.join(os.environ.get('INTERNAL_MODEL_STORAGE'), f"{model_name}.ckpt")

        model = benchmarking.get_model(
            num_labels=config_models.get("num_classes"),
            model_name=config_models.get("name"),
            model_type=config_models.get("type"),
            model_weights_path=config_models.get("weights_path", None),
            model_task=config_models.get("task")
        )

        # Decode base64 image string and convert to torch tensor
        image_bytes = base64.b64decode(body.image)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        transformation = transforms.Compose([
            transforms.Resize(config_models.get("input_dimensionality")[1:]),  # Resize BEFORE converting to tensor
            transforms.ToTensor(),  # Convert to tensor AFTER resizing
        ])

        x: torch.Tensor = transformation(image).unsqueeze(0).to(device)
        labels = model(x).argmax(1)
        y: torch.Tensor = torch.nn.functional.one_hot(labels, num_classes=1000)
        ###################### Attack ######################
        parameters = {param.id: param.default for param in body.attack.parameters}
        atk_cnf = EAF.get_config(
            class_id=body.attack.id,
            model=model,
            task=Task.Classification,
            device=device,
            **parameters
        )

        atk = EAF.create(
            body.attack.id,
            atk_cnf
        )

        start = time.time()
        x_adv = atk.generate(x=x, y=y).detach()
        pert = x_adv - x
        end = time.time()
        y_adv = model(x_adv).argmax(-1)

        ###################### Analysing the results ######################
        # Prepare image data to return
        buffered = io.BytesIO()
        transforms.ToPILImage()(x_adv[0]).save(buffered, format="PNG")
        adv_img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        buffered = io.BytesIO()
        pert_image = transforms.ToPILImage()(pert[0])
        pert_image.save(buffered, format="PNG")
        pert_image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        # Execute the attack and get results
        ssim = StructuralSimilarityIndexMeasure().to(device)

        return SingleAttackOutput(
            x=body.image,
            adv_perturbation=pert_image_base64,
            x_adv=adv_img_base64,
            original_prediction=str(y.argmax(-1).item()),
            adversarial_prediction=str(y_adv.item()),
            confidence={
                "adversarial": [],
                "original": []
            },
            advance_metrics={
                "ssim": ssim(x, x_adv).item(),
                "distance": torch.norm(pert, p=1).item(),
                "executionTime": end - start
            })

    except Exception as e:
        logging.error(f"Unexpected error during attack: {str(e)}")

        return Error(
            code=500,
            message=f"Unexpected error during attack"
        )
