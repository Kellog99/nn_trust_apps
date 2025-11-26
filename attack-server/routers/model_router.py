import json
import logging
import os
import pickle
import shutil
import tempfile
import zipfile
from pathlib import Path

import torch
from fastapi import APIRouter, Response, Query
from fastapi import UploadFile, HTTPException
from fastapi.responses import JSONResponse

from lib.models import Models, Error
from lib.validator import json_safety_check

router = APIRouter(prefix="/model", tags=["datasets and models"])


@router.get("/getModels", response_model=Models, responses={
    '400': {'model': Error},
    '404': {'model': Error},
    '500': {'model': Error},
})
def get_models():
    """
    Get all the models of the TITANN backend.
    """
    try:
        with open(os.environ.get("TIMM_MODELS_JSON_PATH")) as f:
            config = json.load(f)
            MODELS = config["timm_models"]
            for n in MODELS:
                n["type"] = "timm"
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

                # Construct the corresponding JSON file path
                json_path = item_path.replace(".pth", ".json")

                # Check if JSON file exists
                if not os.path.isfile(json_path):
                    raise FileNotFoundError(f"JSON file not found for model: {item}. Expected: {json_path}")

                # Read the JSON file
                with open(json_path, 'r') as json_file:
                    model_info = json.load(json_file)

                # Create model entry with base info and extend with JSON data
                model_entry = {"name": Path(item).stem, "type": "saved_model"}
                merged_model_info = model_info | model_entry
                models.append(merged_model_info)

        if len(models) == 0:
            logging.info("No uploaded models found.")

        models.extend(MODELS)
        if len(models) == 0:
            logging.info("No models found.")
            return Response(status_code=404,
                            content=Error(code=404, message="No models found.").model_dump_json())

        models = Models(models=models)
        return Response(status_code=200,
                        content=models.model_dump_json())

    except Exception as e:
        logging.error(f"An error occurred during models reading from disk: {e}")
        return Response(status_code=500,
                        content=Error(code=500,
                                      message=f"An error occurred durign models reading from disk.").model_dump_json())


# --- Model Upload (Check Phase) ---
@router.post("/upload/check", response_model=None, responses={
    '400': {'model': Error},
    '500': {'model': Error},
})
async def upload_model_check(file: UploadFile):
    """
    Step 1: Upload a model package as a ZIP file, validate its structure and content.
    The ZIP must contain:
    - One model file (.pt, .pth, .pkl, .pickle)
    - One metadata file (.json)
    The extracted content is stored in a temporary folder for the next step.
    """

    # Settings and Config Import
    try:
        # Load config to get existing model names, although model existence check is moved to 'proceed'
        with open(os.environ.get("TIMM_MODELS_JSON_PATH", "/tmp/models_config.json")) as f:
            config = json.load(f)
            # MODELS = config.get("timm_models", {}) # Not strictly needed here, but keeping structure

    except Exception as e:
        logging.error(f"Unexpected error during config import: {str(e)}")
        # Using HTTPException here for consistency with other 'check' endpoint if desired,
        # but sticking to Response for compatibility with original code's return type.
        return Response(
            status_code=500,
            content=Error(code=500, message=f"Internal server error during config import.").model_dump_json())

    # 1. Validate file type is ZIP
    if not file.filename or not file.filename.endswith('.zip'):
        logging.error("Invalid file type. File must be a ZIP archive")
        return Response(
            status_code=400,
            content=Error(code=400, message="Invalid file type. File must be a ZIP archive").model_dump_json()
        )

    # 2. Check file size
    MAX_FILE_SIZE = int(os.environ.get("MAX_MODEL_SIZE_UPLOAD", 5000 * 1024 * 1024))
    if file.size and file.size > 5000 * 1024 * 1024:
        limit_mb = MAX_FILE_SIZE // (1024 * 1024)
        logging.error(f"File size exceeds {limit_mb}MB limit")
        return Response(
            status_code=400,
            content=Error(code=400, message=f"File size exceeds {limit_mb}MB limit").model_dump_json()
        )

    temp_dir = None
    model_name = Path(file.filename).stem  # Base name without .zip

    try:
        # Create temporary directory for processing
        # Use model_name in prefix to help locate it in 'proceed'
        temp_dir = tempfile.mkdtemp(prefix=f"titann_model_check_{model_name}_")
        zip_path = os.path.join(temp_dir, file.filename)

        # 3. Save uploaded ZIP file
        with open(zip_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 4. Extract ZIP contents
        extract_dir = os.path.join(temp_dir, model_name)  # Extract to a subdir named after the model
        os.makedirs(extract_dir, exist_ok=True)

        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Security check: validate no path traversal in ZIP
                for member in zip_ref.namelist():
                    if member.startswith('/') or '..' in member or os.path.isabs(member):
                        return Response(
                            status_code=400,
                            content=Error(code=400,
                                          message="Invalid ZIP content - path traversal detected").model_dump_json()
                        )

                zip_ref.extractall(extract_dir)
        except zipfile.BadZipFile:
            return Response(
                status_code=400,
                content=Error(code=400, message="Invalid or corrupted ZIP file").model_dump_json()
            )

        # 5. Find model and metadata files
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

        # 6. Validate exactly one model and one JSON file
        if len(model_files) != 1:
            return Response(
                status_code=400,
                content=Error(code=400,
                              message=f"ZIP must contain exactly one model file. Found {len(model_files)}").model_dump_json()
            )

        if len(json_files) != 1:
            return Response(
                status_code=400,
                content=Error(code=400,
                              message=f"ZIP must contain exactly one JSON metadata file. Found {len(json_files)}").model_dump_json()
            )

        model_path = model_files[0]
        json_path = json_files[0]

        # Check if the primary model name matches the zip name (optional but good practice)
        if Path(model_path).stem != model_name:
            logging.warning(f"Model filename '{Path(model_path).name}' stem does not match zip stem '{model_name}'")

        # 7. Validate and load the model file (partial loading for check)
        model = None
        try:
            if model_path.endswith(('.pt', '.pth')):
                # Use map_location='cpu' and weights_only=False for general check
                model = torch.load(model_path, weights_only=False)
            elif model_path.endswith(('.pkl', '.pickle')):
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)

        except Exception as e:
            logging.error(f"Failed to load model from {model_path}: {str(e)}")
            return Response(
                status_code=400,
                content=Error(code=400, message=f"Invalid or corrupted model file: {str(e)}").model_dump_json()
            )

        # 8. Validate it's a valid PyTorch model
        if not isinstance(model, torch.nn.Module):
            return Response(
                status_code=400,
                content=Error(code=400, message="File does not contain a valid PyTorch model").model_dump_json()
            )

        # 9. Process and validate JSON metadata
        logging.info("Starting JSON metadata check...")
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                raw_json_metadata = f.read()

            # Validate JSON
            checked_metadata = json_safety_check(raw_json_metadata)

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

        # Successfully validated. Return the metadata and leave files in temp_dir.
        logging.info(f"Model check successful. Extracted to {extract_dir}. Ready for upload.")

        # Return the validated JSON metadata
        return JSONResponse(status_code=200, content=checked_metadata)

    except Exception as e:
        # If an error occurs before temp_dir is created, this handles it.
        logging.error(f"Unexpected error during model check: {str(e)}")
        # Cleanup in case the exception happened after temp_dir was created but before 'finally'
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except Exception as cleanup_e:
                logging.error(f"Failed to cleanup temp dir after error: {cleanup_e}")

        return Response(
            status_code=500,
            content=Error(code=500, message=f"Internal server error during model check: {e}").model_dump_json()
        )

    finally:
        # Note: We skip cleanup here. Cleanup is done in the 'proceed' step.
        pass


# --- Model Upload (Proceed Phase) ---
@router.get("/upload", response_model=None, responses={
    '400': {'model': Error},
    '409': {'model': Error},
    '500': {'model': Error},
})
async def upload_model_proceed(
        model_name: str = Query(..., description="The name of the model (stem of the original ZIP file)")) -> Response:
    """
    Step 2: Proceed to upload the validated model. Moves the folder from tmp to permanent storage.
    """
    temp_dir_to_cleanup = None  # Used to track the parent temp directory for final cleanup

    try:
        # Load config to get existing model names
        with open(os.environ.get("TIMM_MODELS_JSON_PATH", "/tmp/models_config.json")) as f:
            config = json.load(f)
            MODELS = config.get("timm_models", {})

    except Exception as e:
        logging.error(f"Unexpected error during config import: {str(e)}")
        return Response(
            status_code=500,
            content=Error(code=500, message=f"Internal server error during config import.").model_dump_json())

    try:
        # 1. Find the model in tmp directories
        tmp_base = tempfile.gettempdir()
        extracted_content_path = None

        # Search for the temporary directory created in the check step
        for tmp_dir in os.listdir(tmp_base):
            if tmp_dir.startswith(f"titann_model_check_{model_name}_"):
                tmp_path = os.path.join(tmp_base, tmp_dir)
                potential_extracted_dir = os.path.join(tmp_path, model_name)
                if os.path.exists(potential_extracted_dir):
                    extracted_content_path = potential_extracted_dir
                    temp_dir_to_cleanup = tmp_path
                    break

        if not extracted_content_path:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{model_name}' not found in temporary storage. Please run check first."
            )

        # 2. Re-locate the model and JSON files from the temporary extracted folder
        model_extensions = ['.pt', '.pth', '.pkl', '.pickle']
        model_files = []
        json_files = []

        for root, dirs, files in os.walk(extracted_content_path):
            for filename in files:
                file_path = os.path.join(root, filename)
                if any(filename.endswith(ext) for ext in model_extensions):
                    model_files.append(file_path)
                elif filename.endswith('.json'):
                    json_files.append(file_path)

        # We rely on the 'check' step for validation, but assert file paths are available
        if len(model_files) != 1 or len(json_files) != 1:
            logging.error(f"Temp directory for '{model_name}' is corrupted or invalid.")
            # Treat as 500 or 400 depending on cause, but 400 is safer if files were removed by another process
            raise HTTPException(status_code=400, detail="Temporary model files are missing or corrupted.")

        model_path = model_files[0]
        json_path = json_files[0]
        model_filename = os.path.basename(model_path)

        # 3. Check for existing model (409 Conflict)
        model_storage_path = os.environ.get('INTERNAL_MODEL_STORAGE', '/tmp/model_storage')

        if os.path.exists(os.path.join(model_storage_path, model_filename)) or model_name in MODELS:
            return Response(
                status_code=409,
                content=Error(code=409, message=f"Model '{model_name}' already exists").model_dump_json()
            )

        # 4. Copy the model file to permanent storage
        dest_model_path = os.path.join(model_storage_path, model_filename)
        logging.info(f"Copying model from {model_path} to {dest_model_path}")
        os.makedirs(model_storage_path, exist_ok=True)  # Ensure storage dir exists
        shutil.copy2(model_path, dest_model_path)
        os.chmod(dest_model_path, 0o600)  # Read/write for owner only

        # 5. Load and save the validated JSON metadata
        dest_json_path = os.path.join(model_storage_path, f"{model_name}.json")

        # Read the validated JSON from the temp folder
        with open(json_path, 'r', encoding='utf-8') as f:
            raw_json_metadata = f.read()

        # Re-parse/re-check the metadata before saving for safety and to get dict form
        # We assume json_safety_check can handle both string and dict input, 
        # but since we read the file here, it's a string.
        json_metadata = json_safety_check(raw_json_metadata)

        # Save the JSON metadata
        with open(dest_json_path, "w", encoding="utf-8") as f:
            json.dump(json_metadata, f, ensure_ascii=False, indent=2)
        os.chmod(dest_json_path, 0o600)  # Read/write for owner only

        logging.info(f"Successfully uploaded model '{model_name}' from file '{model_filename}'")
        logging.info(f"JSON metadata saved as '{model_name}.json'")

        # Successful upload response
        return Response(status_code=200)

    except HTTPException:
        # Re-raise explicit HTTP exceptions (e.g., model not found in tmp)
        raise
    except Exception as e:
        logging.error(f"Unexpected error during model upload: {str(e)}")
        return Response(
            status_code=500,
            content=Error(code=500, message=f"Internal server error during model upload: {e}").model_dump_json()
        )

    finally:
        # 6. Cleanup temporary files
        if temp_dir_to_cleanup and os.path.exists(temp_dir_to_cleanup):
            try:
                shutil.rmtree(temp_dir_to_cleanup)
                logging.info(f"Cleaned up temporary directory: {temp_dir_to_cleanup}")
            except Exception as e:
                logging.error(f"Failed to cleanup temporary directory {temp_dir_to_cleanup}: {str(e)}")

# Combine all new and old router definitions for completeness if needed in one file
# from backendserver.bslib.singletons.dataset import Datasets # Placeholder if needed
# router.post("/upload_folder/check", ...) # Existing /upload_folder/check endpoint
# router.get("/upload_folder", ...)     # Existing /upload_folder endpoint
