from fastapi import APIRouter
from pathlib import Path
import importlib
from typing import List, Union, Optional
from lib.models import Error, Metric, Progress, Result, JobConfig
from lib import models
from typing import Annotated
from fastapi import Depends, FastAPI, HTTPException, Query, Response
from sqlmodel import Field, Session, SQLModel, create_engine, select
from database.models import BenchmarkJob, AttackJob
import os,json
import logging
from sqlalchemy.exc import NoResultFound, MultipleResultsFound


import sys
import base64
import io
from PIL import Image

print(sys.path)
celery_utils = importlib.import_module("attack-server.celery_src.utils")
celery_tasks = importlib.import_module("attack-server.celery_src.celery_worker")
benchmarking = importlib.import_module("benchmarking")


router = APIRouter(prefix="/job", tags=["jobs management", "jobs utils"])
DB_CONFIG_FILE = Path(__file__).parent.parent / "resources" / "config.json"

# --------------- DB --------------------

with open(DB_CONFIG_FILE) as f:
    config = json.load(f)
    sqlite_file_name = config["db_name"]
    sqlite_url = config["db_url"]

connect_args = {"check_same_thread": False}
engine = create_engine(sqlite_url, connect_args=connect_args)

def get_session():
    with Session(engine) as session:
        yield session

SessionDep = Annotated[Session, Depends(get_session)]


def update_attack_job(session, job_id, progress):
    statement = select(AttackJob).where(AttackJob.id == job_id)
    res = session.exec(statement).one()
    res.progress = progress
    session.add(res)
    session.commit()


# ----------------- SERVICES --------------------------

@router.get("/getJobsId", response_model=List[BenchmarkJob], responses={
    '400': {'model': Error},
    '404': {'model': Error},
    '500': {'model': Error},
}, tags=["jobs utils"])
def get_jobs_id(session : SessionDep) -> Union[List[BenchmarkJob], Error]:
    """
    Get all running jobs in the TITANN backend.
    """
    try:
        jobs = session.exec(select(BenchmarkJob).limit(100)).all()
        if len(jobs)>0:
            return jobs
        else:
            return Response(status_code=404, content=Error(code=404, message="No jobs running!").model_dump_json())
    except Exception as e:
        logging.error(f"Unexpected error during get jobs: {str(e)}")
        return Response(
                status_code=500,
                content=Error(code=500, message=f"Unexpected error during get jobs").model_dump_json())

@router.get("/getAttackJobsId", response_model=List[AttackJob], responses={
    '400': {'model': Error},
    '404': {'model': Error},
    '500': {'model': Error},
}, tags=["jobs utils"])
def get_attack_jobs_id(session : SessionDep) -> Union[List[AttackJob], Error]:
    """
    Get all running jobs in the TITANN backend.
    """
    try:
        jobs = session.exec(select(AttackJob).limit(100)).all()
        if len(jobs)>0:
            return jobs
        else:
            return Response(status_code=404, content=Error(code=404, message="No jobs running!").model_dump_json())
    except Exception as e:
        logging.error(f"Unexpected error during get jobs: {str(e)}")
        return Response(
                status_code=500,
                content=Error(code=500, message=f"Unexpected error during get jobs").model_dump_json())

@router.get("/getProgress", response_model=Progress, responses={
    '400': {'model': Error},
    '500': {'model': Error},
}, tags=["jobs management"])
def get_job_progress(id: str, session : SessionDep) -> Union[Progress, Error]:
    """
    Get a TITANN benchmark job progress.
    """
    try:
        try:
            job = session.exec(select(BenchmarkJob).where(BenchmarkJob.id == id).limit(100)).one()
        except NoResultFound:
            return Response(status_code=404, 
                            content=Error(code=404, message=f"No job found with id {id}").model_dump_json())
        except MultipleResultsFound:
            return Response(status_code=500, 
                            content=Error(code=500, message=f"Multiple jobs found with id {id}. Check database!").model_dump_json())
        return Response(
            status_code=200,
            content=Progress(
                progress=job.progress,
                is_over=job.is_over
            ).model_dump_json()
        )
    except Exception as e:
        logging.error(f"Unexpected error during get progress: {str(e)}")
        return Response(
                status_code=500,
                content=Error(code=500, message=f"Unexpected error during get progress").model_dump_json())

@router.get("/getResult", response_model=Result, responses={
    '400': {'model': Error},
    '404': {'model': Error},
    '500': {'model': Error},
}, tags=["jobs management"])
def get_job_result(id: str, session : SessionDep) -> Union[Result, Error]:
    """
    Get a TITANN benchmark job result.
    """
    try:
        try:
            job = session.exec(select(BenchmarkJob).where(BenchmarkJob.id == id).where(BenchmarkJob.is_over==True).limit(100)).one()
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

@router.get("/restart", response_model=None, responses={
    '400': {'model': Error},
    '409': {'model': Error},
    '500': {'model': Error},
}, tags=["jobs management"])
def restart_job(id: str) -> Optional[Error]:
    """
    Restart a new TITANN benchmark job from last checkpoint (last performed attack).
    """
    pass


@router.post("/benchmark", response_model=None, responses={
    '400': {'model': Error},
    '500': {'model': Error},
}, tags=["jobs management"])
def start_job(body: benchmarking.BenchmarkConfigModel, session : SessionDep) -> Optional[Error]:
    """
    Start a new TITANN benchmark job.
    """
    try:
        with session.begin():
            new_job = BenchmarkJob(progress=0.0, 
                          dataset=body.datasets[0].name, 
                          model=body.models[0].name,
                          is_over=False)
            session.add(new_job)
            session.flush()
            benchmark_task = celery_tasks.benchmarking_task.delay(body.dict())
            #print(f"Completed")
            #update_attack_job(session, new_job.id, progress=1.0)
            
        #session.refresh(new_job)
        return Response(status_code=200,content=json.dumps({"task_id":benchmark_task.id}), 
            media_type="application/json"
        )

    except Exception as e:
        logging.error(f"Unexpected error during job start: {str(e)}")
        return Response(
                status_code=500,
                content=Error(code=500, message=f"Unexpected error during job start").model_dump_json())

@router.get("/stop", response_model=None, responses={
    '400': {'model': Error},
    '409': {'model': Error},
    '500': {'model': Error},
}, tags=["jobs management"])
def stop_job(id: str) -> Optional[Error]:
    """
    Stop a TITANN benchmark job.
    """
    pass


@router.post("/single_attack", response_model=None, responses={
    '400': {'model': Error},
    '500': {'model': Error},
}, tags=["jobs management"])
async def start_singleattack_job(body: models.AttackConfig, session : SessionDep) -> Optional[Error]:
    """
    Start a new TITANN benchmark job.
    """
    try:
        with session.begin():
            # ================================================
            # TODO: Put “start job” logic here!
            #       - Perform any setup or validations.
            #       - If anything fails, raise an exception
            #         to roll back the transaction.
            # ================================================
            new_job = AttackJob(progress=0.1, is_over=False)
            session.add(new_job)
            session.flush() 
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
            print(f"Completed")
            update_attack_job(session, new_job.id, progress=1.0)

        session.refresh(new_job)
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