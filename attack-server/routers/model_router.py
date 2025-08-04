from fastapi import APIRouter, UploadFile, Response
from typing import Union, Optional
from lib.models import Models, Error
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Optional
import torch
import pickle
from fastapi import UploadFile, HTTPException
import logging
import json

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
    try:
        with open(os.path.join("attack-server","resources","config.json")) as f:
                config = json.load(f)
                MODELS = config["timm_models"]
    except Exception as e:
        # Handle unexpected errors
        logging.error(f"Unexpected error during config import: {str(e)}")
        return Response(
                    status_code=500,
                    content=Error(code=500, message=f"Internal server error during config import.").model_dump_json())
    try:
        models = []
        models_root_dir = os.environ.get("INTERNAL_MODEL_STORAGE")
        for item in os.listdir(models_root_dir):
            item_path = os.path.join(models_root_dir, item)
            if os.path.isfile(item_path):
                logging.info(f"Found a model: {item_path}")
                models.append(Path(item).stem)
        if len(models)==0:
            logging.info("No uploaded models found.")
        else:
            router.state.uploaded_models = models

        models.extend(MODELS)
        if len(models)==0:
            logging.info("No models found.")
            return Response(status_code=404, 
                            content=Error(code=404, message="No models found.").model_dump_json())

        models = Models(names=models)
        return Response(status_code=200, 
                        content=models.model_dump_json())

    except Exception as e:
        logging.error(f"An error occurred during models reading from disk: {e}")
        return Response(status_code=500, 
                        content=Error(code=500, message=f"An error occurred durign models reading from disk.").model_dump_json())

@router.post("/upload", response_model=None, responses={
    '400': {'model': Error},
    '409': {'model': Error},
    '500': {'model': Error},
})
def upload_model(file: UploadFile) -> Optional[Error]:
    """
    Upload a model directly to the TITANN backend.
    Accepts PyTorch model files (.pt, .pth, .pkl, .pickle).
    """
    # Settings
    try:
        with open(os.path.join("attack-server","resources","config.json")) as f:
                config = json.load(f)
                MODELS = config["timm_models"]
                MAX_FILE_SIZE = config["max_model_size"]
    except Exception as e:
        # Handle unexpected errors
        logging.error(f"Unexpected error during config import: {str(e)}")
        return Response(
                    status_code=500,
                    content=Error(code=500, message=f"Internal server error during config import.").model_dump_json())
    
    # Validate file type
    expected_extensions = ['.pt', '.pth', '.pkl', '.pickle']
    if not any(file.filename.endswith(ext) for ext in expected_extensions):
        logging.error(f"Invalid file type. File must be one of: {', '.join(expected_extensions)}")
        return Response(
            status_code=400,
            content=Error(code=400, message=f"Invalid file type. File must be one of: {', '.join(expected_extensions)}").model_dump_json()
        )
    
    # Check file size (adjust limit as needed, e.g., 500MB)
    MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
    if file.size and file.size > MAX_FILE_SIZE:
        logging.error(f"File size exceeds {MAX_FILE_SIZE // (1024*1024)}MB limit")
        return Response(
            status_code=400,
            content=Error(code=400, message=f"File size exceeds {MAX_FILE_SIZE // (1024*1024)}MB limit").model_dump_json()
        )
    
    temp_dir = None
    
    try:
        # Create temporary directory for processing
        temp_dir = tempfile.mkdtemp(prefix="titann_upload_")
        model_path = os.path.join(temp_dir, file.filename)
        
        # Save uploaded file
        with open(model_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Validate filename doesn't contain path traversal
        safe_filename = os.path.basename(file.filename)
        if safe_filename != file.filename or '..' in file.filename or file.filename.startswith('/'):
            return Response(
                status_code=400,
                content=Error(code=400, message="Invalid filename - path traversal detected").model_dump_json()
            )
        
        # Validate and load the model file
        model = None
        try:
            if model_path.endswith(('.pt', '.pth')):
                # Load PyTorch model file
                model = torch.load(model_path, weights_only=False)
            elif model_path.endswith(('.pkl', '.pickle')):
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
            else:
                return Response(
                    status_code=400,
                    content=Error(code=400, message="Unsupported file format").model_dump_json()
                )
            
        except Exception as e:
            logging.error(f"Failed to load model from {model_path}: {str(e)}")
            return Response(
                    status_code=400,
                    content=Error(code=400, message=f"Invalid or corrupted model file: {str(e)}").model_dump_json()
                )
        
        # Validate it's a valid PyTorch model
        is_valid_model = False
        if isinstance(model, torch.nn.Module):
            is_valid_model = True
        if not is_valid_model:
            return Response(
                    status_code=400,
                    content=Error(code=400, message="File does not contain a valid PyTorch model").model_dump_json()
                )
        
        # Check for existing model
        model_name = Path(file.filename).stem
        model_storage_path = os.environ.get('INTERNAL_MODEL_STORAGE')
        
        if os.path.exists(os.path.join(model_storage_path,safe_filename)) or model_name in MODELS:
            return Response(
                    status_code=409,
                    content=Error(code=409, message=f"Model '{model_name}' already exists").model_dump_json()
                )
        
        # Copy the model file to permanent storage
        dest_path = os.path.join(model_storage_path, safe_filename)
        shutil.copy2(model_path, dest_path)
        
        # Additional security: Set restrictive permissions on the stored file
        os.chmod(dest_path, 0o600)  # Read/write for owner only
        
        logging.info(f"Successfully uploaded model '{model_name}' from file '{safe_filename}'")
        return Response(status_code=200)
        
    except Exception as e:
        # Handle unexpected errors
        logging.error(f"Unexpected error during model upload: {str(e)}")
        return Response(
                    status_code=500,
                    content=Error(code=500, message=f"Internal server error during model upload.")
                )
    
    finally:
        # Cleanup temporary files
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                logging.error(f"Failed to cleanup temporary directory: {str(e)}")
