from fastapi import APIRouter
from typing import List, Union, Optional
from lib.models import Error, Metric, Progress, Result, JobConfig
from typing import Annotated
from fastapi import Depends, FastAPI, HTTPException, Query, Response
from sqlmodel import Field, Session, SQLModel, create_engine, select
from database.models import Job
import os,json
import logging
from sqlalchemy.exc import NoResultFound, MultipleResultsFound

router = APIRouter(prefix="/job", tags=["jobs management", "jobs utils"])

####### DB #######
    
with open(os.path.join("attack-server","resources","config.json")) as f:
    config = json.load(f)
    sqlite_file_name = config["db_name"]
    sqlite_url = config["db_url"]

connect_args = {"check_same_thread": False}
engine = create_engine(sqlite_url, connect_args=connect_args)

def get_session():
    with Session(engine) as session:
        yield session

SessionDep = Annotated[Session, Depends(get_session)]


####### SERVICES #######
@router.get("/getJobsId", response_model=List[Job], responses={
    '400': {'model': Error},
    '404': {'model': Error},
    '500': {'model': Error},
}, tags=["jobs utils"])
def get_jobs_id(session : SessionDep) -> Union[List[Job], Error]:
    """
    Get all running jobs in the TITANN backend.
    """
    try:
        jobs = session.exec(select(Job).where(Job.is_over == False).limit(100)).all()
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
            job = session.exec(select(Job).where(Job.id == id).limit(100)).one()
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
            job = session.exec(select(Job).where(Job.id == id).where(Job.is_over==True).limit(100)).one()
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


@router.post("/start", response_model=None, responses={
    '400': {'model': Error},
    '500': {'model': Error},
}, tags=["jobs management"])
def start_job(body: JobConfig, session : SessionDep) -> Optional[Error]:
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
            new_job = Job(progress=0.43, 
                          dataset=body.dataset, 
                          model=body.model,
                          is_over=False)
            session.add(new_job)
            session.flush()  
            
        session.refresh(new_job)
        return Response(status_code=200)

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
