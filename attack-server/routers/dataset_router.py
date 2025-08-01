from fastapi import APIRouter, UploadFile, Response
from typing import Union, Optional
import os
import shutil
import zipfile
import logging
from lib.models import Datasets, Error

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
    try:
        datasets = []
        dataset_root_dir = os.environ.get("INTERNAL_DS_STORAGE")
        for item in os.listdir(dataset_root_dir):
            item_path = os.path.join(dataset_root_dir, item)
            if os.path.isdir(item_path):
                logging.info(f"Found a dataset: {item_path}")
                datasets.append(item_path)
        if len(datasets)==0:
            logging.error("No datasets found.")
            return Response(status_code=404, 
                            content=Error(code=404, message="No datasets found.").model_dump_json())

        datasets = Datasets(names=datasets)
        return Response(status_code=200, 
                        content=datasets.model_dump_json())

    except Exception as e:
        logging.error(f"An error occurred datasets reading from disk: {e}")
        return Response(status_code=500, 
                        content=Error(code=500, message=f"An error occurred datasets reading from disk: {e}").model_dump_json())

@router.post("/upload", response_model=None, responses={
    '400': {'model': Error},
    '409': {'model': Error},
    '500': {'model': Error},
})
def upload_dataset(file: UploadFile) -> Optional[Error]:
    """
    Upload a dataset to the TITANN backend.
    """
    try:
        if not file.filename.endswith(".zip"):
            logging.error("Error: Only.zip files are allowed.")
            return Response(status_code=400,
                            content=Error(code=400,detail="Only .zip files are allowed.").model_dump_json())
        
        UPLOAD_DIRECTORY = os.environ.get('INTERNAL_DS_STORAGE')
        if not UPLOAD_DIRECTORY:
            logging.error("Error: No internal dataset storage is specified in the environment.")
            return Response(status_code=500,
                            content=Error(code=500,detail="Upload directory not configured.").model_dump_json())
        
        file_path = os.path.join(UPLOAD_DIRECTORY, file.filename)
    except Exception as e:
        logging.error(f"An error occurred before the zip copy and extraction: {e}")
        return Response(status_code=500,
                        content=Error(code=500, message=f"An error occurred before the zip copy and extraction: {e}").model_dump_json())
    
    try:
        # Save the uploaded zip file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logging.info("File saved.")

        # Extract the zip file
        extract_to_path = UPLOAD_DIRECTORY
        os.makedirs(extract_to_path, exist_ok=True)
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to_path)
        logging.info("File extracted.")

        return Response(status_code=200)
        
    except zipfile.BadZipFile:
        logging.error("Invalid or corrupted zip file.")
        return Response(status_code=400, 
                        content=Error(code=400,detail="Invalid or corrupted zip file.").model_dump_json())
    except PermissionError:
        logging.error("Permission denied when accessing upload directory.")
        return Response(status_code=403, 
                        content=Error(code=403,detail="Permission denied when accessing upload directory.").model_dump_json())
    except Exception as e:
        logging.error(f"Failed to process file: {str(e)}")
        return Response(status_code=500, 
                        content=Error(code=500, detail=f"Failed to process file: {str(e)}").model_dump_json())
    finally:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            logging.error(f"Exception occurred in the removal of the .zip: {e}")
            pass  
