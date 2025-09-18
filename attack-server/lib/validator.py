from fastapi import UploadFile, File, HTTPException, status
import json
import magic 
import os
import logging
from pathlib import Path
from pydantic import ValidationError
from lib.models import ModelConfig

MAX_BYTES = int(os.environ.get('MAX_MODEL_JSON_SIZE_UPLOAD')) * 1024
CHUNK_SIZE = 4096
ALLOWED_MIMES = {"application/json"}


async def read_up_to(upload_file: UploadFile, max_bytes: int) -> bytes:
    await upload_file.seek(0)
    total = 0
    parts = []
    while True:
        chunk = await upload_file.read(CHUNK_SIZE)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            await upload_file.close()
            logging.error(f"File exceeded size limit of {max_bytes} bytes")
            raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                                detail=f"File exceeded size limit of {max_bytes} bytes")
        parts.append(chunk)
    return b"".join(parts)


async def json_safety_check(metadata: UploadFile = File(...)) -> dict:
    """
    Checks if a JSON is secure with various strategies and saves it to disk.

    """

    logging.info("Checking header...")
    header_mime = (metadata.content_type or "").lower()
    
    logging.info("Checking max dimension...")
    raw = await read_up_to(metadata, MAX_BYTES)

    try:
        detected = magic.from_buffer(raw, mime=True)  # should return 'application/json'
    except Exception:
        logging.error("No mime type detected")
        detected = None

    stripped = raw.lstrip()[:4] 
    likely_json_by_chars = False
    if stripped:
        first = chr(stripped[0]) if isinstance(stripped[0], int) else stripped[0]
        if first in ("{", "[", '"'):  # object, array or a raw string (still JSON)
            likely_json_by_chars = True

    mime_ok = (detected in ALLOWED_MIMES) or (header_mime in ALLOWED_MIMES) or likely_json_by_chars
    if not mime_ok:
        await metadata.close()
        logging.error(f"Uploaded file does not look like JSON (detected={detected}, header={header_mime})")
        raise HTTPException(status_code=400, detail=f"Uploaded file does not look like JSON (detected={detected}, header={header_mime})")

    # 6) parse JSON to be sure it's valid JSON
    try:
        decoded = raw.decode("utf-8")
    except UnicodeDecodeError:
        await metadata.close()
        logging.error("File must be UTF-8 encoded text")
        raise HTTPException(status_code=400, detail="File must be UTF-8 encoded text")

    try:
        obj = json.loads(decoded)
    except json.JSONDecodeError:
        await metadata.close()
        logging.error("Invalid JSON")
        raise HTTPException(status_code=400, detail="Invalid JSON")

    # 7) final checks you may want: ensure top-level type is object/array
    if not isinstance(obj, (dict, list)):
        await metadata.close()
        logging.error("JSON top-level must be an object or array")
        raise HTTPException(status_code=400, detail="JSON top-level must be an object or array")

    await metadata.close()
    
    try:
        validated = ModelConfig.model_validate(obj)  
    except ValidationError as e:
        await metadata.close()
        logging.error(f"Metadata validation failed: {e}")
        raise HTTPException(status_code=422, detail=e.errors())

    return obj
