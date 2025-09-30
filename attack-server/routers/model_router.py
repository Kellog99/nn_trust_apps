from fastapi import APIRouter, UploadFile, Response
from fastapi.responses import JSONResponse
from typing import Union, Optional
from lib.models import Models, Error
from lib.validator import json_safety_check
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
        with open(os.environ.get("TIMM_MODELS_JSON_PATH")) as f:
                config = json.load(f)
                MODELS = config["timm_models"]
                for n in MODELS:
                    n["mode"]="timm"
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
            if os.path.isfile(item_path) and item_path.endswith(".pth"):
                logging.info(f"Found a model: {item_path}")
                models.append({"name":Path(item).stem,"mode":"saved_model"})
        if len(models)==0:
            logging.info("No uploaded models found.")

        models.extend(MODELS)
        if len(models)==0:
            logging.info("No models found.")
            return Response(status_code=404, 
                            content=Error(code=404, message="No models found.").model_dump_json())

        models = Models(models=models)
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
async def upload_model(file: UploadFile) -> Optional[Error]:
    """
    Upload a model package as a ZIP file to the TITANN backend.
    The ZIP must contain:
    - One model file (.pt, .pth, .pkl, .pickle)
    - One metadata file (.json)
    """
    # Settings
    try:
        with open(os.environ.get("TIMM_MODELS_JSON_PATH")) as f:
            config = json.load(f)
            MODELS = config["timm_models"]
            
    except Exception as e:
        logging.error(f"Unexpected error during config import: {str(e)}")
        return Response(
            status_code=500,
            content=Error(code=500, message=f"Internal server error during config import.").model_dump_json())
    
    # Validate file type is ZIP
    if not file.filename.endswith('.zip'):
        logging.error(f"Invalid file type. File must be a ZIP archive")
        return Response(
            status_code=400,
            content=Error(code=400, message="Invalid file type. File must be a ZIP archive").model_dump_json()
        )
    MAX_FILE_SIZE = int(os.environ.get("MAX_MODEL_SIZE_UPLOAD", 5000 * 1024 * 1024))

    # Check file size
    if file.size and file.size > 5000 * 1024 * 1024:
        logging.error(f"File size exceeds {MAX_FILE_SIZE // (1024*1024)}MB limit")
        return Response(
            status_code=400,
            content=Error(code=400, message=f"File size exceeds {MAX_FILE_SIZE // (1024*1024)}MB limit").model_dump_json()
        )

    temp_dir = None
    
    try:
        # Create temporary directory for processing
        temp_dir = tempfile.mkdtemp(prefix="titann_upload_")
        zip_path = os.path.join(temp_dir, file.filename)
        
        # Save uploaded ZIP file
        with open(zip_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Extract ZIP contents
        extract_dir = os.path.join(temp_dir, "extracted")
        os.makedirs(extract_dir, exist_ok=True)
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Security check: validate no path traversal in ZIP
                for member in zip_ref.namelist():
                    if member.startswith('/') or '..' in member or os.path.isabs(member):
                        return Response(
                            status_code=400,
                            content=Error(code=400, message="Invalid ZIP content - path traversal detected").model_dump_json()
                        )
                
                zip_ref.extractall(extract_dir)
        except zipfile.BadZipFile:
            return Response(
                status_code=400,
                content=Error(code=400, message="Invalid or corrupted ZIP file").model_dump_json()
            )
        
        # Find model and metadata files in extracted contents
        model_extensions = ['.pt', '.pth', '.pkl', '.pickle']
        model_files = []
        json_files = []
        
        for root, dirs, files in os.walk(extract_dir):
            for filename in files:
                file_path = os.path.join(root, filename)
                if any(filename.endswith(ext) for ext in model_extensions):
                    model_files.append(file_path)
                elif filename.endswith('.json'):
                    json_files.append(file_path)
        
        # Validate exactly one model and one JSON file
        if len(model_files) != 1:
            return Response(
                status_code=400,
                content=Error(code=400, message=f"ZIP must contain exactly one model file. Found {len(model_files)}").model_dump_json()
            )
        
        if len(json_files) != 1:
            return Response(
                status_code=400,
                content=Error(code=400, message=f"ZIP must contain exactly one JSON metadata file. Found {len(json_files)}").model_dump_json()
            )
        
        model_path = model_files[0]
        json_path = json_files[0]
        model_filename = os.path.basename(model_path)
        
        
        # Validate and load the model file
        model = None
        try:
            if model_path.endswith(('.pt', '.pth')):
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
        if not isinstance(model, torch.nn.Module):
            return Response(
                status_code=400,
                content=Error(code=400, message="File does not contain a valid PyTorch model").model_dump_json()
            )
        
        # Check for existing model
        model_name = Path(model_filename).stem
        model_storage_path = os.environ.get('INTERNAL_MODEL_STORAGE')
        
        if os.path.exists(os.path.join(model_storage_path, model_filename)) or model_name in MODELS:
            return Response(
                status_code=409,
                content=Error(code=409, message=f"Model '{model_name}' already exists").model_dump_json()
            )
        
        # Process and validate JSON metadata
        logging.info("Starting JSON metadata check and storage...")
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                json_metadata = json.load(f)
            
            # Validate JSON (assuming json_safety_check is a validation function)
            # If you need async validation, adapt accordingly
            json_metadata = json_safety_check(json_metadata)
            
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON metadata: {str(e)}")
            return Response(
                status_code=400,
                content=Error(code=400, message=f"Invalid JSON metadata: {str(e)}").model_dump_json()
            )
        except Exception as e:
            logging.error(f"Failed to process JSON metadata: {str(e)}")
            return Response(
                status_code=400,
                content=Error(code=400, message=f"Failed to process JSON metadata: {str(e)}").model_dump_json()
            )
        
        # Copy the model file to permanent storage
        dest_model_path = os.path.join(model_storage_path, model_filename)
        shutil.copy2(model_path, dest_model_path)
        os.chmod(dest_model_path, 0o600)  # Read/write for owner only
        
        # Save the JSON metadata
        dest_json_path = os.path.join(model_storage_path, f"{model_name}.json")
        with open(dest_json_path, "w", encoding="utf-8") as f:
            json.dump(json_metadata, f, ensure_ascii=False, indent=2)
        os.chmod(dest_json_path, 0o600)  # Read/write for owner only
        
        logging.info(f"Successfully uploaded model '{model_name}' from file '{model_filename}'")
        logging.info(f"JSON metadata saved as '{model_name}.json'")

        return JSONResponse(status_code=200, content=json_metadata)
        
    except Exception as e:
        logging.error(f"Unexpected error during model upload: {str(e)}")
        return Response(
            status_code=500,
            content=Error(code=500, message=f"Internal server error during model upload.").model_dump_json()
        )
    
    finally:
        # Cleanup temporary files
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                logging.error(f"Failed to cleanup temporary directory: {str(e)}")
