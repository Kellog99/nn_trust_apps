from fastapi import APIRouter
from fastapi.responses import JSONResponse
import importlib
from typing import Union, Optional
from lib.models import Error, ExecutionConfig
from lib.disk_reader import find_model_and_task_dir, collect_dataset_aggregates_with_info
from lib.attack_utils import build_benchmark_dict
from lib import models
from fastapi import Response, Query, Body
import json
import logging
import base64
import io
import os
from PIL import Image
from routers.dataset_router import get_datasets
from routers.model_router import get_models
import ray
from pathlib import Path
import PIL
import torch
from nn_trust.attack import EvasionAttackFactory, EvasionAttackConfig
from nn_trust.core import ModelAdapter, Task, Knowledge
import logging
from torchvision import transforms
import PIL


benchmarking = importlib.import_module("benchmarking")

router = APIRouter(prefix="/job", tags=["jobs management", "jobs utils"])

# Setting up number of actors and executor
num_actors = int(os.environ.get("RAY_NUM_ACTORS",1))
modules = os.environ.get("RAY_PY_MODULES",None)      
if modules:
    ray.init(ignore_reinit_error=True,runtime_env={
        "py_modules": [modules]
    })
executor = benchmarking.benchmark_utils.executor.RayActorPoolExecutor(num_actors=num_actors)

# ------------------ UTILITY --------------------------

def check_model_and_dataset_in_running_jobs(job, d, m):
    """
    Checks if the specified dataset and model to perform a benchmark on are already running.
    """
    return job.get('dataset') == d and job.get('model') == m

def check_attack_already_launched(attacks, req_attacks):
    """
    Checks if one or more specified attacks is already running in the backend.
    """
    return any(item in attacks for item in req_attacks), set(attacks) & set(req_attacks)

def has_aggregate(task_dir: str, dataset: str) -> bool:
    """Return True if the given task_dir/<dataset> contains any 'aggregate.json' file."""
    task_path = Path(task_dir).expanduser().resolve()
    dataset_dir = task_path / dataset
    if not dataset_dir.is_dir():
        return False
    return any(p.name == "aggregate.json" for p in dataset_dir.rglob("aggregate.json"))
    
# ----------------- SERVICES --------------------------

# --- Progress --- #
@router.get("/getJobs")
def get_jobs(id : str = Query(None)):
    """
    Get all running benchmark jobs in the TITANN backend.
    """
    try:
        tasks = ray.get(executor.tracker.list_tasks.remote())
        if id:
            id = id.replace(" ","")
            return {k: v for k, v in tasks.items() if v["benchmark_id"]==id}
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
                     dataset : str = Query(None),
                     model : str = Query(None)):
    """
    Get a TITANN benchmark report job result.
    """
    try:
    
        model_dir, task_dir = find_model_and_task_dir(os.environ.get("BENCHMARK_OUTPUT_DIR"),dataset,model,id)
        benchmark_id = task_dir.split(os.sep)[-1]
        tasks = {k:v for k,v in ray.get(executor.tracker.list_tasks.remote()).items() if v["benchmark_id"]==benchmark_id}

        if not tasks:
            logging.error(f"Benchmark {benchmark_id} not found")
            return Response(
                    status_code=404,
                    content=Error(code=404, message=f"Benchmark {benchmark_id} not found").model_dump_json())
        
        completed_tasks = []
        for _,v in tasks.items():
            num_tasks = v["num_tasks"]
            if v["benchmark_id"]==benchmark_id and v["status"]=="completed" and v["progress"]==100:
                completed_tasks.append(v)
    
        if len(completed_tasks)==0 or len(completed_tasks)<num_tasks:
            logging.error(f"Benchmark {benchmark_id} not finished yet")
            return Response(
                    status_code=409,
                    content=Error(code=409, message=f"Benchmark {benchmark_id} not finished yet").model_dump_json())
 
        benchmarking.postprocess_benchmark_run_results(task_dir)
        with open(os.path.join(model_dir,'info.json'), "r", encoding="utf-8") as f:
            info = json.load(f)
        with open(os.path.join(model_dir,'aggregate_statistics.json'), "r", encoding="utf-8") as f:
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
                            statistics[entry] = json.load(sf)
                    except Exception as e:
                        logging.warning(f"Could not load statistics.json in '{entry_path}': {e}")
        
        return {
            "info":info,
            "metrics": aggregate,
            "attacks": statistics
        }
    
    except Exception as e:
        logging.error(f"Unexpected error during get result: {str(e)}")
        return Response(
                status_code=500,
                content=Error(code=500, message=f"Unexpected error during get result").model_dump_json())
 
@router.get("/benchmark/getResult")
def get_jobs_results(dataset : str = Query(...), task : str = Query(None)):
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
        
        completed_tasks = []

        for _,v in tasks.items():
            num_tasks = v["num_tasks"]
            if v["status"]=="completed" and v["progress"]==100:
                completed_tasks.append(v)
                task_dir = os.path.join(os.environ.get("BENCHMARK_OUTPUT_DIR"),v["benchmark_id"])
                if not has_aggregate(task_dir,dataset):
                    benchmarking.postprocess_benchmark_run_results(task_dir)

        if len(completed_tasks)==0 or len(completed_tasks)<num_tasks:
            logging.error(f"No benchmark is finished yet")
            return Response(
                    status_code=409,
                    content=Error(code=409, message=f"No benchmark is finished yet").model_dump_json())
        
        results = collect_dataset_aggregates_with_info(
            base_dir=os.environ.get("BENCHMARK_OUTPUT_DIR"),
            dataset=dataset,
        )
        return build_benchmark_dict(results,dataset,"classification")
    
    except Exception as e:
        logging.error(f"Unexpected error during get result: {str(e)}")
        return Response(
                status_code=500,
                content=Error(code=500, message=f"Unexpected error during get result").model_dump_json())


# --- Start --- #
@router.post("/start", response_model=str)
async def start_benchmark_job(body: ExecutionConfig = Body(...)) -> Union[str,Error]:
    """
    Start a new TITANN benchmark job.
    """
    try:
        dataset_name = body.dataset
        model_name = body.model
        m_response = get_models()
        d_response = get_datasets()
        model_response = json.loads(m_response.body.decode('utf-8'))["models"]

        if m_response.status_code!=200 or d_response.status_code!=200:
            logging.error("Model or Dataset repository is empty. Check repository.")
            return Response(
                status_code=404,
                content=Error(code=404, 
                            message="Model or Dataset repository is empty. Check repository.").model_dump_json())

        model = [m["name"] for m in model_response if m["name"]==model_name]
        model_type = [m["type"] for m in model_response if m["name"]==model_name]
        dataset = [d for d in json.loads(d_response.body.decode('utf-8'))["names"] if d==dataset_name]

        if len(model)>1 or len(model_type)>1:
            logging.error(f"The provided model string {model_name} has had more than one match. Check model repository")
            return Response(
                status_code=409,
                content=Error(code=409, 
                              message=f"The provided model string {model_name} has had more than one match. Check model repository").model_dump_json())
        
        if len(model)==0 or len(model_type)==0:
            logging.error(f"No model found: {model_name}")
            return Response(
                status_code=404,
                content=Error(code=404, 
                              message=f"No model found: {model_name}").model_dump_json())
        
        if len(dataset)>1:
            logging.error(f"The provided dataset string {dataset_name} has had more than one match. Check dataset repository")
            return Response(
                status_code=409,
                content=Error(code=409, 
                              message=f"The provided dataset string {dataset_name} has had more than one match. Check dataset repository").model_dump_json())
        
        if len(dataset)==0:
            logging.error(f"No dataset found: {dataset_name}")
            return Response(
                status_code=404,
                content=Error(code=404, 
                              message=f"No dataset found: {dataset_name}").model_dump_json())
        
        file_path = os.path.join(os.environ.get('INTERNAL_MODEL_STORAGE'),f"{model_name}.json")

        if model_type[0]=="saved_model":
            if os.path.exists(file_path) and os.path.isfile(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    config_models = json.load(f)  # load into dict
                    config_models["type"]="saved_model"
                logging.info("Loaded JSON metadata")
            else:
                logging.error(f"Can't find model metadata in the repository for model:{model_name}")
                return Response(
                    status_code=404,
                    content=Error(code=404, 
                                message=f"Can't find model metadata in the repository for model:{model_name}").model_dump_json())
        elif model_type[0]=="timm":
            model_metadata = None
            for m in model_response:
                if m["name"]==model_name:
                    model_metadata = m
            if model_metadata:
                config_models = model_metadata
            else:
                logging.error("An error has occurred for timm model metadata retrieval")
                return Response(
                    status_code=404,
                    content=Error(code=404, 
                                message="An error has occurred for timm model metadata retrieval").model_dump_json())
            
        file_path = os.path.join(os.environ.get('INTERNAL_DS_STORAGE'),dataset_name,f"{dataset_name}.json")
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

        if model_type[0]=="saved_model":
            config_models["weights_path"] = os.path.join(os.environ.get('INTERNAL_MODEL_STORAGE'),f"{model_name}.pth")

        config_dataset["source_path"] = os.path.join(dataset_name,"data")

        benchmark_config['datasets'] = [config_dataset]
        benchmark_config['models'] = [config_models]
        benchmark_config['attacks'] = body.attacks
        benchmark_config['evaluation'] = {}
        benchmark_config["evaluation"]["statistics"] = body.metrics
        task_id = benchmarking.benchmark(benchmark_config,executor)
        return JSONResponse(status_code=200,content=task_id)

    except Exception as e:
        logging.error(f"Unexpected error during job start: {str(e)}")
        return Response(
                status_code=500,
                content=Error(code=500, message=f"Unexpected error during job start").model_dump_json())


# --- Single attack --- #
@router.post("/attack")
async def start_singleattack_job(body: models.AttackConfig) -> Optional[Error]:
    """
    Start a new TITANN attack on single image job.
    """
    try:
        
        # Decode base64 image string and convert to torch tensor
        image_bytes = base64.b64decode(body.image)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Get the model
        model_name = body.model_name
        m_response = get_models()
        model_response = json.loads(m_response.body.decode('utf-8'))["models"]

        if m_response.status_code!=200:
            logging.error("Model repository is empty. Check repository.")
            return Response(
                status_code=404,
                content=Error(code=404, 
                            message="Model or Dataset repository is empty. Check repository.").model_dump_json())
        
        model = [m["name"] for m in model_response if m["name"]==model_name]
        model_type = [m["type"] for m in model_response if m["name"]==model_name]

        if len(model)>1 or len(model_type)>1:
            logging.error(f"The provided model string {model_name} has had more than one match. Check model repository")
            return Response(
                status_code=409,
                content=Error(code=409, 
                              message=f"The provided model string {model_name} has had more than one match. Check model repository").model_dump_json())
        
        if len(model)==0 or len(model_type)==0:
            logging.error(f"No model found: {model_name}")
            return Response(
                status_code=404,
                content=Error(code=404, 
                              message=f"No model found: {model_name}").model_dump_json())
        
        file_path = os.path.join(os.environ.get('INTERNAL_MODEL_STORAGE'),f"{model_name}.json")

        if model_type[0]=="saved_model":
            if os.path.exists(file_path) and os.path.isfile(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    config_models = json.load(f)  # load into dict
                    config_models["type"]="saved_model"
                logging.info("Loaded JSON metadata")
            else:
                logging.error(f"Can't find model metadata in the repository for model:{model_name}")
                return Response(
                    status_code=404,
                    content=Error(code=404, 
                                message=f"Can't find model metadata in the repository for model:{model_name}").model_dump_json())
        elif model_type[0]=="timm":
            model_metadata = None
            for m in model_response:
                if m["name"]==model_name:
                    model_metadata = m
            if model_metadata:
                config_models = model_metadata
            else:
                logging.error("An error has occurred for timm model metadata retrieval")
                return Response(
                    status_code=404,
                    content=Error(code=404, 
                                message="An error has occurred for timm model metadata retrieval").model_dump_json())
        
        if model_type[0]=="saved_model":
            config_models["weights_path"] = os.path.join(os.environ.get('INTERNAL_MODEL_STORAGE'),f"{model_name}.pth")

        model_ad = benchmarking.get_model(
            model_name=config_models.get("name"),
            model_type=config_models.get("type"),
            model_weights_path=config_models.get("weights_path", None),
            model_task=config_models.get("task")
        )
        
        def run_attack(model_ad, img: PIL.Image,attack_name:str,epsilon:float,p:float,max_iters:int):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logging.info(f"Running attack on {device}")

            img = transforms.ToTensor()(img).unsqueeze(0).to(device)
            
            model_ad.model.eval()
            model_ad.model.to(device)

            cnf = EvasionAttackFactory.get_config(
                attack_type=attack_name,
                model=model_ad,
                task=Task.Classification,
                device=device,
                verbose=True,
                epsilon=epsilon,
                p=p,
                max_iters=max_iters
                )

            atk = EvasionAttackFactory.create_attack(attack_type=attack_name, config=cnf)
            labels = model_ad(img).argmax(1)
            labels_ohe = torch.nn.functional.one_hot(labels, num_classes=1000)
            x_adv = atk.generate(x=img, y=labels_ohe)
            labels_adv = model_ad(x_adv).argmax(1)
            return transforms.ToPILImage()(x_adv[0]), labels.item(), labels_adv.item()
        
        adv_img, y, y_adv = run_attack(
            model_ad=model_ad,
            img=image,
            attack_name=body.attack_name,
            p=body.p,
            epsilon=body.epsilon,
            max_iters=body.max_iters
        )
        
        # Prepare image data to return
        buffered = io.BytesIO()
        adv_img.save(buffered, format="PNG")
        adv_img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        #TODO:adapt output 
        result_data = {
            "status": "success",
            "adv_img": adv_img_base64,
            "y": y,
            "y_adv": y_adv
        }

        return Response(
            status_code=200, 
            content=json.dumps(result_data), 
            media_type="application/json"
        )

    except Exception as e:
        logging.error(f"Unexpected error during attack: {str(e)}")
        return Response(
                status_code=500,
                content=Error(code=500, message=f"Unexpected error during attack").model_dump_json())