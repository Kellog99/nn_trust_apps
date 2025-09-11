from fastapi import APIRouter
import importlib
from typing import List, Union, Optional
from lib.models import AttackJob, BenchmarkJob, Error, Metric, Result
from lib.extractors import CeleryRedisExtractor
from lib import models
from fastapi import Response, Query, Body
import json
import logging
from sqlalchemy.exc import NoResultFound, MultipleResultsFound
from celery_src.celery_worker import celery
from celery.result import AsyncResult
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

def get_job_metadata_from_id(id:str):
    """
    This function gets metadata of a Celery Task running in the result backend.

    Args: id - Celery task id
    
    """
    result = AsyncResult(id, app=celery)
    logging.info(f"Tracking task: {id}")
    logging.info("-" * 50)
    if not result.ready():
        if result.state == 'PROGRESS':
            meta = result.info
        else:
            meta = None
            logging.info(f"\rTask state: {result.state}", end='')
        return meta
    elif result.ready():
        return result.info
    else:
        raise Exception("An error occurred during job metadata extraction from id")
    
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

@router.get("/benchmark/getJobs", response_model=List[BenchmarkJob], responses={
    '400': {'model': Error},
    '404': {'model': Error},
    '500': {'model': Error},
}, tags=["jobs utils"])
def get_jobs() -> Union[List[Union[BenchmarkJob,AttackJob]], Error]:
    """

    Get all running benchmark jobs in the TITANN backend.

    """
    try:
        #TODO: Redis host and port should become ENV VARIABLES!
        extractor = CeleryRedisExtractor()
        task_ids = extractor.extract()
        jobs = []
        for id in tqdm(task_ids,desc="Fetching jobs..."):
            meta = get_job_metadata_from_id(id)
            if meta:
                jobs.append(BenchmarkJob(
                            id=id,
                            progress= meta.get('progress', 0),
                            current_attack = meta.get('current_attack', None),
                            is_over = meta.get('is_over', False),
                            dataset = meta.get('dataset', '-'),
                            model = meta.get('model','-'),
                            attack_progress = meta.get('attack_progress',0.0),
                            all_attacks = meta.get('all_attacks',[])
                ))
        if len(jobs)>0:
            return Response(status_code=200, content=json.dumps([j.dict() for j in jobs]), media_type="application/json")
        else:
            return Response(status_code=404, content=Error(code=404, message="No jobs running!").model_dump_json())
    except Exception as e:
        logging.error(f"Unexpected error during get jobs: {str(e)}")
        return Response(
                status_code=500,
                content=Error(code=500, message=f"Unexpected error during get jobs").model_dump_json())
    

@router.get("/attack/getJobProgress", response_model=AttackJob, responses={
    '400': {'model': Error},
    '404': {'model': Error},
    '500': {'model': Error},
}, tags=["jobs utils"])
def get_attack_job_progress(id : str) -> Union[AttackJob, Error]:
    """
    Get a TITANN attack job progress.
    """
    #try:
    #    jobs = []
    #    
    #    #jobs = session.exec(select(AttackJob).limit(100)).all()
    #    if len(jobs)>0:
    #        return jobs
    #    else:
    #        return Response(status_code=404, content=Error(code=404, message="No jobs running!").model_dump_json())
    #except Exception as e:
    #    logging.error(f"Unexpected error during get jobs: {str(e)}")
    #    return Response(
    #            status_code=500,
    #            content=Error(code=500, message=f"Unexpected error during get jobs").model_dump_json())
    pass

@router.get("/benchmark/getJobProgress", response_model=BenchmarkJob, responses={
    '400': {'model': Error},
    '500': {'model': Error},
}, tags=["jobs management"])
def get_benchmark_job_progress(id: str):
    """
    Get a TITANN benchmark job progress.
    """
    try:
        meta = get_job_metadata_from_id(id)
        if meta:
            logging.info(f"Metadata for job {id} found succesfully")
            job = BenchmarkJob(id=id,
                progress= meta.get('progress', 0),
                current_attack = meta.get('current_attack', None),
                is_over = meta.get('is_over', False),
                dataset = meta.get('dataset', '-'),
                model = meta.get('model','-'),
                attack_progress = meta.get('attack_progress',0.0),
                all_attacks = meta.get('all_attacks',[])
                )
            return job
            
        else:
            logging.error(f"No job found with id {id}")
            return Response(status_code=404, content=Error(code=404, message=f"No job found with id {id}").model_dump_json())
    except Exception as e:
        logging.error(f"Unexpected error during get job progress: {str(e)}")
        return Response(
                status_code=500,
                content=Error(code=500, message=f"Unexpected error during get job progress").model_dump_json())

@router.get("/getResult", response_model=Result, responses={
    '400': {'model': Error},
    '404': {'model': Error},
    '500': {'model': Error},
}, tags=["jobs management"])
def get_job_result(id: str) -> Union[Result, Error]:
    """
    Get a TITANN benchmark job result.
    """
    try:
        try:
            #TODO
            pass
        except NoResultFound:
            return Response(status_code=404, 
                            content=Error(code=404, message=f"No terminated job found with id {id}").model_dump_json())
        except MultipleResultsFound:
            return Response(status_code=500, 
                            content=Error(code=500, message=f"Multiple jobs found with id {id}. Check database!").model_dump_json())
        # ================================================
        # TODO: Put “aggregate output on disk" logic here!
        # ================================================
        return Result(
            metrics=[Metric(values=0.8)]
        )
    except Exception as e:
        logging.error(f"Unexpected error during get result: {str(e)}")
        return Response(
                status_code=500,
                content=Error(code=500, message=f"Unexpected error during get result").model_dump_json())

@router.post("/benchmark", response_model=None, responses={
    '400': {'model': Error},
    '500': {'model': Error},
    '409': {'model': Error},
}, tags=["jobs management"])
def start_bechmark_job(dataset_name : str = Query(...), 
                       model_name : str = Query(...), 
                       body: benchmarking.BenchmarkConfigModel = Body(...)) -> Optional[Error]:
    """
    Start a new TITANN benchmark job.
    """
    try:
        job_response = get_jobs()
        
        active_jobs = json.loads(job_response.body.decode('utf-8'))
        active_attacks = []

        if job_response.status_code==200:
            for j in active_jobs:
                active_attacks = (j['all_attacks'])
                requested_attacks = [atk['name'] for atk in body.dict()['attacks']]
                already_launched = check_attack_already_launched(active_attacks,requested_attacks)
                if (check_model_and_dataset_in_running_jobs(j,dataset_name,model_name)
                    and already_launched[0]):
                    logging.error(f"The requested attacks {already_launched[1]} is already running on this combination of dataset and model: {dataset_name} - {model_name}")
                    return Response(
                        status_code=409,
                        content=Error(code=409, 
                                    message=f"One of the requested attacks {requested_attacks} is already running on this combination of dataset and model: {dataset_name} - {model_name}").model_dump_json())
        
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
        model_type = [m["mode"] for m in model_response if m["name"]==model_name]
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
        benchmark_config = body.dict()
        config_dataset = benchmark_config['datasets']
        config_models = benchmark_config['models']

        if len(config_dataset)>1 or len(config_models)>1:
            logging.error(f"Benchmark can be launched only on one dataset and model at a time.")
            return Response(
                status_code=400,
                content=Error(code=400, 
                              message=f"Benchmark can be launched only on one dataset and model at a time.").model_dump_json())

        config_dataset[0]["name"] = dataset_name
        config_models[0]["name"] = model[0]
        config_models[0]["type"] = model_type[0]

        if model_type[0]=="saved_model":
            config_models[0]["weights_path"] = os.path.join(os.environ.get('INTERNAL_MODEL_STORAGE'),f"{model_name}.pth")
        
        benchmark_task = celery_tasks.benchmarking_task.delay(benchmark_config)
        return Response(status_code=200,content=json.dumps({"task_id":benchmark_task.id}), 
            media_type="application/json"
        )
    except Exception as e:
        logging.error(f"Unexpected error during job start: {str(e)}")
        return Response(
                status_code=500,
                content=Error(code=500, message=f"Unexpected error during job start").model_dump_json())

@router.post("/single_attack", response_model=None, responses={
    '400': {'model': Error},
    '500': {'model': Error},
}, tags=["jobs management"])
async def start_singleattack_job(body: models.AttackConfig) -> Optional[Error]:
    """
    Start a new TITANN benchmark job.
    """
#    try:
#        
#        # Decode base64 image string and convert to torch tensor
#        image_bytes = base64.b64decode(body.image)
#        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
#        adv_img, y, y_adv = celery_utils.run_attack(
#            img=image,
#            attack_name=body.attack_name,
#            p=body.p,
#            epsilon=body.epsilon,
#            max_iters=body.max_iters
#        )
#        
#        # Prepare image data to return
#        buffered = io.BytesIO()
#        adv_img.save(buffered, format="PNG")
#        adv_img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
#        result_data = {
#            "status": "success",
#            "adv_img": adv_img_base64,
#            "y": y,
#            "y_adv": y_adv
#        }
#
#        return Response(
#            status_code=200, 
#            content=json.dumps(result_data), 
#            media_type="application/json"
#        )
#
#    except Exception as e:
#        logging.error(f"Unexpected error during job start: {str(e)}")
#        return Response(
#                status_code=500,
#                content=Error(code=500, message=f"Unexpected error during job start").model_dump_json())
    pass

@router.get("/stop", responses={
    '400': {'model': Error},
    '404': {'model': Error},
    '409': {'model': Error},
    '500': {'model': Error},
}, tags=["jobs management"])
def stop_job(id: str) -> Optional[Error]:
    """
    Stop a TITANN benchmark job.
    """
    try:
        redis_client = redis.StrictRedis(host=os.environ.get('REDIS_HOST','localhost'), 
                                        port=os.environ.get('REDIS_PORT',6379), 
                                        db=0, 
                                        decode_responses=True)
        redis_client.set(f"cancel_flag:{id}", "1")
        key = f'celery-task-meta-{id}'
        current_task = redis_client.get(key)
        if current_task is None:
            logging.error(f"Task with id {id} not found")
            return Response(
                status_code=404,
                content=Error(code=404, message=f"Task with id {id} not found").model_dump_json())
        else:
            data = json.loads(current_task)
            data['status'] = 'KILLED'  
            new_value = json.dumps(data)
            redis_client.set(key, new_value)
            logging.info(f"Status updated in key {key}")
            return Response(status_code=200, content=f"Task {id} stopped correctly")
        
    except Exception as e:
        logging.error(f"Unexpected error during stop job: {str(e)}")
        return Response(
                status_code=500,
                content=Error(code=500, message=f"Unexpected error during stop job").model_dump_json())

#@router.post("/single_attack", response_model=None, responses={
#    '400': {'model': Error},
#    '500': {'model': Error},
#}, tags=["jobs management"])
#async def start_singleattack_job(body: models.AttackConfig, session : SessionDep) -> Optional[Error]:
#    """
#    Start a new TITANN benchmark job.
#    """
#    try:
#        with session.begin():
#            # ================================================
#            # TODO: Put “start job” logic here!
#            #       - Perform any setup or validations.
#            #       - If anything fails, raise an exception
#            #         to roll back the transaction.
#            # ================================================
#            new_job = AttackJob(progress=0.1, is_over=False)
#            session.add(new_job)
#            session.flush() 
#            # Decode base64 image string and convert to torch tensor
#            image_bytes = base64.b64decode(body.image)
#            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
#            adv_img, y, y_adv = celery_utils.run_attack(
#                img=image,
#                attack_name=body.attack_name,
#                p=body.p,
#                epsilon=body.epsilon,
#                max_iters=body.max_iters
#            )
#            print(f"Completed")
#            update_attack_job(session, new_job.id, progress=1.0)
#
#        session.refresh(new_job)
#        # Prepare image data to return
#        buffered = io.BytesIO()
#        adv_img.save(buffered, format="PNG")
#        adv_img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
#        result_data = {
#            "status": "success",
#            "adv_img": adv_img_base64,
#            "y": y,
#            "y_adv": y_adv
#        }
#
#        return Response(
#            status_code=200, 
#            content=json.dumps(result_data), 
#            media_type="application/json"
#        )
#
#    except Exception as e:
#        logging.error(f"Unexpected error during job start: {str(e)}")
#        return Response(
#                status_code=500,
#                content=Error(code=500, message=f"Unexpected error during job start").model_dump_json())