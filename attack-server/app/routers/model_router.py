from fastapi import APIRouter, UploadFile
from typing import Union, Optional
from app.models import Models, Error

router = APIRouter(prefix="/model", tags=["datasets and models"])

@router.get("/getModels", response_model=Models, responses={
    '400': {'model': Error},
    '404': {'model': Error},
    '500': {'model': Error},
})
def get_models() -> Union[Models, Error]:
    """
    Get all models of the TITANN backend.
    """
    pass

@router.post("/upload", response_model=None, responses={
    '400': {'model': Error},
    '409': {'model': Error},
    '500': {'model': Error},
})
def upload_model(file: UploadFile) -> Optional[Error]:
    """
    Upload a model to the TITANN backend.
    """
    pass
