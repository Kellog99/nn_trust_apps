import json
import logging
import os
import pickle
import shutil
import tempfile
import zipfile
from pathlib import Path
from routers.utils import load_models_metadata_from_repo
import torch
from fastapi import APIRouter, Response, Query
from fastapi import UploadFile, HTTPException
from fastapi.responses import JSONResponse

from lib.model import Error, ModelInfo
from lib.validator import json_safety_check

router = APIRouter(prefix="/model", tags=["datasets and models"])


@router.get("/getModels", responses={
    '400': {'model': Error},
    '404': {'model': Error},
    '500': {'model': Error},
})
def get_models() :
    """
    Get all models of the TITANN backend.
    """
    try:
        with open(os.environ.get("TIMM_MODELS_JSON_PATH")) as f:
                config = json.load(f)
                MODELS = config["timm_models"]
                for n in MODELS:
                    n["type"]="timm"
    except Exception as e:
        # Handle unexpected errors
        logging.error(f"Unexpected error during config import: {str(e)}")
        return Response(
                    status_code=500,
                    content=Error(code=500, message=f"Internal server error during config import.").model_dump_json())
    try:
        models = load_models_metadata_from_repo()
        models.extend(MODELS)
        if len(models)==0:
            logging.info("No models found.")
            return Response(status_code=404, 
                            content=Error(code=404, message="No models found.").model_dump_json())

        return models

    except Exception as e:
        logging.error(f"An error occurred during models reading from disk: {e}")
        return Response(status_code=500, 
                        content=Error(code=500, message=f"An error occurred durign models reading from disk.").model_dump_json())


# --- Model Upload (Check Phase) ---
@router.post("/upload")
async def upload_model(file: UploadFile):
    """
    The ZIP must contain:
    - One model file (.pt, .pth, .pkl, .pickle,.ckpt)
    - One metadata file (.json)
    """
    try:
        if not file.filename.endswith(".zip"):
            logging.error("Error: Only.zip files are allowed.")
            return Response(status_code=400,
                            content=Error(code=400, message="Only .zip files are allowed.").model_dump_json())

        UPLOAD_DIRECTORY = os.environ.get('MODEL_REPO')
        if not UPLOAD_DIRECTORY:
            logging.error("Error: No internal model storrepoage is specified in the environment.")
            return Response(status_code=500,
                            content=Error(code=500, message="Upload directory not configured.").model_dump_json())

        file_path = os.path.join(UPLOAD_DIRECTORY, file.filename)

    except Exception as e:
        logging.error(f"An error occurred before the zip copy and extraction: {e}")
        return Response(status_code=500,
                        content=Error(code=500,
                                      message=f"An error occurred before the zip copy and extraction: {e}").model_dump_json())

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
                        content=Error(code=400, message="Invalid or corrupted zip file.").model_dump_json())
    except PermissionError:
        logging.error("Permission denied when accessing upload directory.")
        return Response(status_code=403,
                        content=Error(code=403,
                                      message="Permission denied when accessing upload directory.").model_dump_json())
    except Exception as e:
        logging.error(f"Failed to process file: {str(e)}")
        return Response(status_code=500,
                        content=Error(code=500, message=f"Failed to process file.").model_dump_json())
    finally:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            logging.error(f"Exception occurred in the removal of the .zip: {e}")
            pass
   