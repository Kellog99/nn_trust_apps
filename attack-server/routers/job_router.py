from fastapi import APIRouter
from fastapi.responses import JSONResponse
import importlib
from typing import List, Union, Optional
from lib.models import AttackJob, BenchmarkJob, Error, ExecutionConfig, Metric, Result
from lib import models
from fastapi import Response, Query, Body
import json
import logging
import base64
import io
import redis
import os
from PIL import Image
from tqdm.auto import tqdm
from routers.dataset_router import get_datasets
from routers.model_router import get_models

celery_utils = importlib.import_module("attack-server.celery_src.utils")
celery_tasks = importlib.import_module("attack-server.celery_src.celery_worker")
benchmarking = importlib.import_module("benchmarking")

router = APIRouter(prefix="/job", tags=["jobs management", "jobs utils"])


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

    
# ----------------- SERVICES --------------------------

@router.get("/getJobs")
def get_jobs(id : str) -> Union[List[BenchmarkJob], Error]:
    """
    Get all running benchmark jobs in the TITANN backend.
    """
    try:
        #TODO: implement optional id filter
        return ""
    except Exception as e:
        logging.error(f"Unexpected error during get jobs: {str(e)}")
        return Response(
                status_code=500,
                content=Error(code=500, message=f"Unexpected error during get jobs").model_dump_json())
    

@router.get("/getResult", response_model=Result)
def get_jobs_results(id: str) -> Union[Result, Error]:
    """
    Get a TITANN benchmark job result.
    """
    try:
        return ""
    except Exception as e:
        logging.error(f"Unexpected error during get result: {str(e)}")
        return Response(
                status_code=500,
                content=Error(code=500, message=f"Unexpected error during get result").model_dump_json())

@router.post("/start", response_model=str)
def start_benchmark_job(body: ExecutionConfig = Body(...)) -> Union[str,Error]:
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
        
        #TODO: dataset and benchmark metadata should not be in the post, but in the repository!
        file_path = os.path.join(os.environ.get('INTERNAL_MODEL_STORAGE'),f"{model_name}.json")

        if model_type[0]=="saved_model":
            if os.path.exists(file_path) and os.path.isfile(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    config_models = json.load(f)  # load into dict
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


        #if len(config_dataset)>1:
        #    logging.error(f"Benchmark can be launched only on one dataset at a time.")
        #    return Response(
        #        status_code=400,
        #        content=Error(code=400, 
        #                      message=f"Benchmark can be launched only on one dataset at a time.").model_dump_json())
        benchmark_config = {}
        benchmark_config['options'] = {
                "load_results": False,
                "overwrite": True,
                "num_images_to_save": -1,
                "save_perturbation": False,
                "gpu": True,
                "output_path": "./benchmark_out",
                "output_format": "report"
                }

        if model_type[0]=="saved_model":
            config_models["weights_path"] = os.path.join(os.environ.get('INTERNAL_MODEL_STORAGE'),f"{model_name}.pth")

        config_dataset["source_path"] = os.path.join(dataset_name,"data")

        benchmark_config['datasets'] = [config_dataset]
        benchmark_config['models'] = [config_models]
        benchmark_config['attacks'] = body.attacks
        benchmark_config['evaluation'] = {}
        benchmark_config["evaluation"]["statistics"] = body.metrics
        
        #TODO:
        benchmarking.benchmark(benchmark_config)
        return JSONResponse(status_code=200,content="id")

    except Exception as e:
        logging.error(f"Unexpected error during job start: {str(e)}")
        return Response(
                status_code=500,
                content=Error(code=500, message=f"Unexpected error during job start").model_dump_json())




@router.post("/attack",responses={
    '400': {'model': Error},
    '500': {'model': Error},
}, tags=["jobs management"])
async def start_singleattack_job(body: models.AttackConfig) -> Optional[Error]:
    """
    Start a new TITANN attack on single image job.
    """
    try:
        
        # Decode base64 image string and convert to torch tensor
        image_bytes = base64.b64decode(body.image)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        #TODO: add custom model support
        adv_img, y, y_adv = celery_utils.run_attack(
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
        #TODO:adapt input, in order to do so, explode run_attack function above also to include model control explicitly
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
        logging.error(f"Unexpected error during job start: {str(e)}")
        return Response(
                status_code=500,
                content=Error(code=500, message=f"Unexpected error during job start").model_dump_json())