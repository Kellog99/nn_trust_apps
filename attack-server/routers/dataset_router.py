import base64
import importlib
import json
import logging
import os
import shutil
import zipfile

from fastapi import APIRouter, UploadFile, Response

from lib.model import Error

benchmarking = importlib.import_module("benchmarking")
router = APIRouter(prefix="/dataset", tags=["datasets and models"])


def find_image(start_dir: str):
    """
    Depth-first search through directories starting at `start_dir`
    to find the first image file. Once found, return the path
    relative to `start_dir`.
    Directories and files are explored in alphabetical order.
    """
    image_exts = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif', '.svg'}
    stack = [start_dir]
    visited = set()

    while stack:
        path = stack.pop()
        try:
            if os.path.islink(path):
                continue

            if os.path.isdir(path):
                real = os.path.realpath(path)
                if real in visited:
                    continue
                visited.add(real)

                try:
                    entries = list(os.scandir(path))
                except PermissionError:
                    continue

                # Sort entries alphabetically by name
                entries.sort(key=lambda e: e.name.lower(), reverse=True)
                # reverse=True because we’re using a stack (LIFO), so we push reversed order
                for entry in entries:
                    stack.append(entry.path)

            else:
                _, ext = os.path.splitext(path)
                if ext.lower() in image_exts:
                    abs_path = os.path.abspath(path)
                    return os.path.join(start_dir.split(os.sep)[-1],os.path.relpath(abs_path, start_dir))
        except Exception:
            continue

    return None

@router.get("/getDatasets")
def get_datasets():
    """
    Get all datasets of the TITANN backend.
    """
    try:
        datasets = []
        dataset_root_dir = os.environ.get("DATASETS_REPO")
        for item in os.listdir(dataset_root_dir):
            item_path = os.path.join(dataset_root_dir, item)
            if os.path.isdir(item_path):
                with open(os.path.join(item_path,f"{item}.json"), "r") as info_file:
                    info_data = json.load(info_file)
                    type_dataset = info_data.get("type_dataset")
                    task = info_data.get("mode")
                datasetObject = benchmarking.get_dataloader(dataset=str(os.path.join(item_path,"data")), 
                                                            subset=1, 
                                                            batch=1, 
                                                            type_dataset=type_dataset, 
                                                            transform = lambda x: x,
                                                            name = item).dataset
                image_path = find_image(item_path)
                def image_to_base64(p):
                    path = os.path.join(dataset_root_dir, p)
                    with open(path, "rb") as img:
                        return base64.b64encode(img.read()).decode("utf-8")
                b64_string = image_to_base64(image_path)
                image_type = "IMAGE_FEATURE"
                logging.info(f"Found a dataset: {item_path}")
                out_item = {
                    "name": item,
                    "size": len(datasetObject),
                    "task": task
                    #"prototype": {
                    #    "datas":[b64_string],
                    #    "type": image_type
                    #}
                }
                datasets.append(out_item)
        if len(datasets)==0:
            logging.error("No datasets found.")
            return Response(status_code=404, 
                            content=Error(code=404, message="No datasets found.").model_dump_json())

        return datasets

    except Exception as e:
        logging.error(f"An error occurred during datasets reading from disk: {e}")
        return Response(status_code=500, 
                        content=Error(code=500, message=f"An error occurred during datasets reading from disk.").model_dump_json())

@router.post("/upload", response_model=None, responses={
    '400': {'model': Error},
    '409': {'model': Error},
    '500': {'model': Error},
})
def upload_dataset(file: UploadFile):
    """
    Upload a dataset to the TITANN backend.
    """
    try:
        if not file.filename.endswith(".zip"):
            logging.error("Error: Only.zip files are allowed.")
            return Response(status_code=400,
                            content=Error(code=400, message="Only .zip files are allowed.").model_dump_json())

        UPLOAD_DIRECTORY = os.environ.get('INTERNAL_DS_STORAGE')
        if not UPLOAD_DIRECTORY:
            logging.error("Error: No internal dataset storage is specified in the environment.")
            return Response(status_code=500,
                            content=Error(code=500, message="Upload directory not configured.").model_dump_json())

        # Check if dataset already exists 
        dataset_name = os.path.splitext(file.filename)[0]
        dataset_folder_path = os.path.join(UPLOAD_DIRECTORY, dataset_name)

        if os.path.exists(dataset_folder_path) and os.path.isdir(dataset_folder_path):
            logging.error(f"Dataset '{dataset_name}' already exists.")
            return Response(status_code=409,
                            content=Error(code=409,
                                          message=f"Dataset - {dataset_name} - already exists.").model_dump_json())

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
