import sys
import os
sys.path.insert(0, os.path.abspath("."))
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import argparse
import os
import uvicorn

app = FastAPI(
    title='Swagger TITANN backend',
    description='This is the TITANN backend.',
    termsOfService='https://swagger.io/terms/',
    contact={'email': 'apiteam@swagger.io'},
    license={
        'name': 'Apache 2.0',
        'url': 'https://www.apache.org/licenses/LICENSE-2.0.html',
    },
    version='1.0.12',
    servers=[{'url': 'https://titann.swagger.io/api/v3'}],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from routers import api_router
# Include routers
app.include_router(api_router)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.get("/")
def root():
    return {"message": "Welcome to the TITANN Job API"}

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ds_storage", "-ds", 
        type=str, 
        default="submodules/data-quality_gui/public/titann/datasets",
        help="Path to internal storage directory (datasets)"
    )
    parser.add_argument(
        "--model_storage", "-ms", 
        type=str, 
        default="submodules/data-quality_gui/public/titann/models",
        help="Path to internal storage directory (models)"
    )
    parser.add_argument(
        "--host", type=str, default="localhost", help="Host to bind the server to (default: localhost)"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to bind the server to (default: 8000)"
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of Uvicorn worker processes"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    os.environ['INTERNAL_DS_STORAGE'] = args.ds_storage
    os.environ['INTERNAL_MODEL_STORAGE'] = args.model_storage
    os.environ['PORT'] = str(args.port)
    os.environ['HOST'] = args.host
    logging.info(f"Starting FastAPI app with internal storage: {args.ds_storage} , {args.model_storage}")
    logging.info(f"Server will run on {args.host}:{args.port} with {args.workers} worker(s)")
    uvicorn.run(
        "app:app",
        host=args.host,
        port=args.port,
        workers=args.workers
    )