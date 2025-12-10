import sys
import os
from pathlib import Path
import logging
import argparse
import ray
# Importing other nn_trust apps scope
sys.path.insert(0, str(Path(__file__).parent.parent))


from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import api_router
import uvicorn



app = FastAPI(
    title='TITANN backend',
    description='This is the TITANN backend.',
#    termsOfService='https://swagger.io/terms/',
#    contact={'email': 'apiteam@swagger.io'},
#    license={
#        'name': 'Apache 2.0',
#        'url': 'https://www.apache.org/licenses/LICENSE-2.0.html',
#    },
#    version='1.0.12',
    servers=[{'url': 'https://titann.swagger.io/api/v3'}],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Include routers
app.include_router(api_router)

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

@app.get("/")
def root():
    return {"message": "Welcome to the TITANN Job API"}

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ds_storage", "-ds", 
        type=str, 
        default="datasets_repo",
        help="Path to internal storage directory (datasets)"
    )
    parser.add_argument(
        "--model_storage", "-ms", 
        type=str, 
        default="models_repo",
        help="Path to internal storage directory (models)"
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to bind the server to (default: localhost)"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to bind the server to (default: 8000)"
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of Uvicorn worker processes"
    )
    parser.add_argument(
        '--benchmark_output_dir', type=str, default='benchmark_results', help='Path to benchmark output directory'
    )
    parser.add_argument(
        '--max_model_size_upload', type=int, default=5000, help='Maximum model file size for upload in MB'
    )
    parser.add_argument(
        '--max_model_json_size_upload', type=int, default=5000, help='Maximum model JSON file size for upload in MB'
    )
    parser.add_argument(
        '--ray_address', type=str, default=None, help='Ray cluster address (e.g., 127.0.0.1:6379). If None, initializes local cluster'
    )
    parser.add_argument(
        '--ray_py_modules', type=str, default=None, help='Path to Python modules to include in Ray runtime environment'
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    # Set all environment variables from arguments
    os.environ['INTERNAL_DS_STORAGE'] = args.ds_storage
    os.environ['INTERNAL_MODEL_STORAGE'] = args.model_storage
    os.environ['BENCHMARK_OUTPUT_DIR'] = args.benchmark_output_dir
    os.environ['PORT'] = str(args.port)
    os.environ['HOST'] = args.host
    os.environ['MAX_MODEL_SIZE_UPLOAD'] = str(args.max_model_size_upload)
    os.environ['MAX_MODEL_JSON_SIZE_UPLOAD'] = str(args.max_model_json_size_upload)
    
    load_dotenv()
    
    logging.info(f"Starting FastAPI app with internal storage: {args.ds_storage}, {args.model_storage}")
    logging.info(f"Benchmark output directory: {args.benchmark_output_dir}")
    logging.info(f"Max upload sizes - Model: {args.max_model_size_upload}MB, JSON: {args.max_model_json_size_upload}MB")
    logging.info(f"Server will run on {args.host}:{args.port} with {args.workers} worker(s)")
    
    # Initialize Ray
    ray_init_kwargs = {
        "ignore_reinit_error": True
    }
    
    # Add address if provided
    if args.ray_address:
        ray_init_kwargs["address"] = args.ray_address
        logging.info(f"Connecting to Ray cluster at: {args.ray_address}")
    else:
        logging.info("Initializing local Ray cluster")
    
    # Add runtime environment with py_modules if provided
    if args.ray_py_modules:
        ray_init_kwargs["runtime_env"] = {
            "py_modules": [args.ray_py_modules]
        }
        logging.info(f"Ray runtime environment py_modules: {args.ray_py_modules}")
    
    try:
        ray.init(**ray_init_kwargs)
        logging.info("Ray initialized successfully")
    except Exception as e:
        logging.error(f"Failed to initialize Ray: {e}")
        raise
    
    uvicorn.run(
        "app:app",
        host=args.host,
        port=args.port,
        workers=args.workers
    )