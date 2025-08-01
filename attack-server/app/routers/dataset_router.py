from fastapi import APIRouter, UploadFile
from typing import Union, Optional
from app.models import Datasets, Error

router = APIRouter(prefix="/dataset", tags=["datasets and models"])

@router.get("/getDatasets", response_model=Datasets, responses={
    '400': {'model': Error},
    '404': {'model': Error},
    '500': {'model': Error},
})
def get_datasets() -> Union[Datasets, Error]:
    """
    Get all datasets of the TITANN backend.
    """
    pass

@router.post("/upload", response_model=None, responses={
    '400': {'model': Error},
    '409': {'model': Error},
    '500': {'model': Error},
})
def upload_dataset(file: UploadFile) -> Optional[Error]:
    """
    Upload a dataset to the TITANN backend.
    """
    pass
