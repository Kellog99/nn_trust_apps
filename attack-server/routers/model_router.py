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

MODELS = [
    "resnet50"
]

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
    # Validate file type
    if not file.filename.endswith('.zip'):
        logging.error("Invalid file type. File must be a .zip")
        return Response(
            status_code=400,
            content=Error(code=400,message="Invalid file type. File must be a .zip")
        )
    
    # Check file size (adjust limit as needed, e.g., 500MB)
    MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
    if file.size and file.size > MAX_FILE_SIZE:
        logging.error(f"File size exceeds {MAX_FILE_SIZE // (1024*1024)}MB limit")
        raise Response(
            status_code=400,
            content=Error(code=400,message=f"File size exceeds {MAX_FILE_SIZE // (1024*1024)}MB limit")
        )
    
    #temp_dir = None
    #extract_dir = None
    #
    #try:
    #    # Create temporary directory for processing
    #    temp_dir = tempfile.mkdtemp(prefix="titann_upload_")
    #    zip_path = os.path.join(temp_dir, file.filename)
    #    
    #    # Save uploaded file
    #    with open(zip_path, "wb") as buffer:
    #        shutil.copyfileobj(file.file, buffer)
    #    
    #    # Validate ZIP file
    #    if not zipfile.is_zipfile(zip_path):
    #        raise HTTPException(
    #            status_code=400,
    #            detail={"message": "Invalid ZIP file format", "code": "INVALID_ZIP"}
    #        )
    #    
    #    # Extract ZIP file
    #    extract_dir = os.path.join(temp_dir, "extracted")
    #    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    #        # Security check: prevent zip bombs, path traversal, and nested directories
    #        for member in zip_ref.namelist():
    #            if member.startswith('/') or '..' in member or '/' in member:
    #                raise HTTPException(
    #                    status_code=400,
    #                    detail={"message": "ZIP must contain only files in root directory (no subdirectories allowed)", "code": "INVALID_ZIP_STRUCTURE"}
    #                )
    #        
    #        zip_ref.extractall(extract_dir)
    #    
    #    # Find and validate model files (directly in the ZIP root)
    #    model_files = []
    #    expected_extensions = ['.pt', '.pth', '.pkl', '.pickle']
    #    
    #    # List files directly in the extracted directory (no subdirectory traversal)
    #    extracted_files = os.listdir(extract_dir)
    #    
    #    for file_name in extracted_files:
    #        file_path = os.path.join(extract_dir, file_name)
    #        if os.path.isfile(file_path) and any(file_name.endswith(ext) for ext in expected_extensions):
    #            model_files.append(file_path)
    #    
    #    if not model_files:
    #        raise HTTPException(
    #            status_code=400,
    #            detail={"message": "No PyTorch model files found in ZIP root", "code": "NO_MODEL_FILES"}
    #        )
    #    
    #    # Validate each model file
    #    valid_models = []
    #    for model_path in model_files:
    #        try:
    #            # Try to load the model
    #            if model_path.endswith(('.pt', '.pth')):
    #                model = torch.load(model_path, map_location='cpu')
    #            elif model_path.endswith(('.pkl', '.pickle')):
    #                with open(model_path, 'rb') as f:
    #                    model = pickle.load(f)
    #            else:
    #                continue
    #            
    #            # Validate it's a PyTorch model
    #            if isinstance(model, torch.nn.Module):
    #                valid_models.append(model_path)
    #            elif isinstance(model, dict) and 'state_dict' in model:
    #                # It's a checkpoint with state_dict
    #                valid_models.append(model_path)
    #            elif isinstance(model, dict) and any(isinstance(v, torch.Tensor) for v in model.values()):
    #                # It's a state_dict directly
    #                valid_models.append(model_path)
    #            
    #        except Exception as e:
    #            # Log the error but continue checking other files
    #            print(f"Failed to load model from {model_path}: {str(e)}")
    #            continue
    #    
    #    if not valid_models:
    #        raise HTTPException(
    #            status_code=400,
    #            detail={"message": "No valid PyTorch models found", "code": "INVALID_MODEL_FORMAT"}
    #        )
    #    
    #    # Check for existing model (if you have a naming convention)
    #    model_name = Path(file.filename).stem
    #    model_storage_path = f"models/{model_name}"  # Adjust path as needed
    #    
    #    if os.path.exists(model_storage_path):
    #        raise HTTPException(
    #            status_code=409,
    #            detail={"message": f"Model '{model_name}' already exists", "code": "MODEL_EXISTS"}
    #        )
    #    
    #    # Move validated model files to permanent storage
    #    os.makedirs(model_storage_path, exist_ok=True)
    #    
    #    # Copy all files from the ZIP root (model files and any supporting files)
    #    for file_name in os.listdir(extract_dir):
    #        src_path = os.path.join(extract_dir, file_name)
    #        if os.path.isfile(src_path):
    #            dest_path = os.path.join(model_storage_path, file_name)
    #            shutil.copy2(src_path, dest_path)
    #    
    #    print(f"Successfully uploaded model '{model_name}' with {len(valid_models)} model file(s)")
    #    return None  # Success case
    #    
    #except HTTPException:
    #    # Re-raise HTTP exceptions
    #    raise
    #except Exception as e:
    #    # Handle unexpected errors
    #    print(f"Unexpected error during model upload: {str(e)}")
    #    raise HTTPException(
    #        status_code=500,
    #        detail={"message": "Internal server error during model upload", "code": "UPLOAD_ERROR"}
    #    )
    #
    #finally:
    #    # Cleanup temporary files
    #    if temp_dir and os.path.exists(temp_dir):
    #        try:
    #            shutil.rmtree(temp_dir)
    #        except Exception as e:
    #            print(f"Failed to cleanup temporary directory: {str(e)}")
    pass
