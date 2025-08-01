from fastapi import APIRouter
from typing import List, Union, Optional
from app.models import Job, Error, Progress, Result, JobConfig

router = APIRouter(prefix="/job", tags=["jobs management", "jobs utils"])

@router.get("/getJobsId", response_model=List[Job], responses={
    '400': {'model': Error},
    '404': {'model': Error},
    '500': {'model': Error},
}, tags=["jobs utils"])
def get_jobs_id() -> Union[List[Job], Error]:
    """
    Get all running jobs in the TITANN backend.
    """
    pass

@router.get("/getProgress", response_model=Progress, responses={
    '400': {'model': Error},
    '500': {'model': Error},
}, tags=["jobs management"])
def get_job_progress(id: str) -> Union[Progress, Error]:
    """
    Get a TITANN benchmark job progress.
    """
    pass

@router.get("/getResult", response_model=Result, responses={
    '400': {'model': Error},
    '404': {'model': Error},
    '500': {'model': Error},
}, tags=["jobs management"])
def get_job_result(id: str) -> Union[Result, Error]:
    """
    Get a TITANN benchmark job result.
    """
    pass

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
def start_job(body: JobConfig) -> Optional[Error]:
    """
    Start a new TITANN benchmark job.
    """
    pass

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
