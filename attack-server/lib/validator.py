from fastapi import UploadFile, File, HTTPException, status
import json
import magic 
import os
import logging
from pathlib import Path
from pydantic import ValidationError
from lib.models import ModelConfig
from typing import Union

MAX_BYTES = int(os.environ.get('MAX_MODEL_JSON_SIZE_UPLOAD',5000)) * 1024
CHUNK_SIZE = 4096
ALLOWED_MIMES = {"application/json"}


def json_safety_check(metadata: Union[str, dict]) -> dict:
    """
    Checks if a JSON is secure with various strategies.
    
    Args:
        metadata: Either a JSON string or a dict object
        
    Returns:
        dict: The validated JSON object
        
    Raises:
        HTTPException: If validation fails
    """
    logging.info("Starting JSON safety check...")
    
    # Handle both string and dict inputs
    if isinstance(metadata, dict):
        obj = metadata
        raw_bytes = json.dumps(metadata).encode('utf-8')
    elif isinstance(metadata, str):
        raw_bytes = metadata.encode('utf-8')
        try:
            obj = json.loads(metadata)
        except json.JSONDecodeError:
            logging.error("Invalid JSON string")
            raise HTTPException(status_code=400, detail="Invalid JSON")
    else:
        logging.error("Input must be a string or dict")
        raise HTTPException(status_code=400, detail="Input must be a JSON string or dict")
    
    # Check size limit
    if len(raw_bytes) > MAX_BYTES:
        logging.error(f"JSON exceeded size limit of {MAX_BYTES} bytes")
        raise HTTPException(status_code=413, 
                          detail=f"JSON exceeded size limit of {MAX_BYTES} bytes")
    
    # Validate top-level type
    if not isinstance(obj, (dict, list)):
        logging.error("JSON top-level must be an object or array")
        raise HTTPException(status_code=400, 
                          detail="JSON top-level must be an object or array")
    
    # Validate against Pydantic model
    try:
        validated = ModelConfig.model_validate(obj)
        logging.info("JSON validation successful")
    except ValidationError as e:
        logging.error(f"Metadata validation failed: {e}")
        raise HTTPException(status_code=422, detail=e.errors())
    
    return obj
