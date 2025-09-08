from fastapi import APIRouter
import importlib
from typing import List, Union, Optional
from lib.models import AttackJob, BenchmarkJob, Error, Metric, Result
from lib.extractors import CeleryRedisExtractor
from lib import models
from fastapi import Response
import json
import logging
from sqlalchemy.exc import NoResultFound, MultipleResultsFound
from celery_src.celery_worker import celery
from celery.result import AsyncResult
import base64
import io
from PIL import Image
from tqdm.auto import tqdm

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
            meta = result.info
            logging.info(f"\rTask state: {result.state}", end='')
        return meta
    elif result.ready():
        return result.info
    else:
        raise Exception("An error occurred during job metadata extraction from id")
    
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
                            last_attack_performed = meta.get('last_attack_performed', None),
                            is_over = meta.get('is_over', False),
                            dataset = meta.get('dataset', '-'),
                            model = meta.get('model','-'),
                            attack_progress = meta.get('attack_progress',0.0)
                ))
        if len(jobs)>0:
            return jobs
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
                last_attack_performed = meta.get('last_attack_performed', None),
                is_over = meta.get('is_over', False),
                dataset = meta.get('dataset', '-'),
                model = meta.get('model','-'),
                attack_progress = meta.get('attack_progress',0.0)
                )
            return job
            
        else:
            logging.error(f"No job found with id {id}")
            return Response(status_code=404, content=Error(code=404, message=f"No job found with id {id}").model_dump_json())
    except Exception as e:
        logging.error(f"Unexpected error during get jobs: {str(e)}")
        return Response(
                status_code=500,
                content=Error(code=500, message=f"Unexpected error during get jobs").model_dump_json())

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
}, tags=["jobs management"])
def start_bechmark_job(body: benchmarking.BenchmarkConfigModel) -> Optional[Error]:
    """
    Start a new TITANN benchmark job.
    """
    try:
        benchmark_task = celery_tasks.benchmarking_task.delay(body.dict())
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
    try:
        
        # Decode base64 image string and convert to torch tensor
        image_bytes = base64.b64decode(body.image)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
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