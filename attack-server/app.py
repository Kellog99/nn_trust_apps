import logging

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from models.main_model import ServerConfig, parsed_argument
from routers import api_router
from routers.info_router import router

########### Environmental variables ###########
args = parsed_argument(ServerConfig)
config = ServerConfig(**vars(args))
###############################################

app = FastAPI(
    title='TITANN backend',
    description='This is the TITANN backend.',
    servers=[{'url': 'https://titann.swagger.io/api/v3'}],
)
app.state.config = config
app.include_router(router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Include routers
app.include_router(api_router)

############### LOGGER ###############
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


#######################################

@app.get("/")
def root():
    return {"message": "Welcome to the TITANN Job API"}


if __name__ == "__main__":
    logging.info(
        f"Starting FastAPI app with internal storage: {config.path_ds_repo}, {config.path_model_repo}"
    )
    logging.info(
        f"Max upload sizes - Model: {config.max_model_size_upload}MB, JSON: {config.max_model_json_size_upload}MB"
    )
    logging.info(
        f"Server will run on {config.host}:{config.port} with {config.workers} worker(s)"
    )


    uvicorn.run(
        "app:app",
        host=config.host,
        port=config.port,
        workers=config.workers
    )




