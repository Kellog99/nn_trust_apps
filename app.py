import json
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from attack_server.routers import api_router
from benchmarking.privacy.loading import ensure_privacy_registries
from models import ServerConfig, parsed_argument

########### Environmental variables ###########
args = parsed_argument(ServerConfig)
if args.configuration_file:
    cnf_path: Path = Path(args.configuration_file).expanduser()
    if cnf_path.exists():
        with open(args.configuration_file, "r") as f:
            file_config = json.load(f)
        config = ServerConfig.model_validate(file_config)
    else:
        raise ValueError("The path to the configuration file does not exist.")
else:
    config = ServerConfig(**vars(args))


###############################################

def create_app() -> FastAPI:
    """
    This function is for using the uvicorn factory function.
    When uvicorn starts multiple workers:
      *  Each worker is a separate process
      *  Each worker imports your code independently
      *  Each worker calls your factory function to create its own app instance
      *  This ensures each worker has properly initialized state
    """
    app = FastAPI(
        title='TITANN backend',
        description='This is the TITANN backend.',
        #servers=[{'url': 'https://titann.swagger.io/api/v3'}],
        # run with: PYTHONPATH=.:submodules/nn_trust:/home/antonio-liguori/.cache/torch/hub/chenyaofo_pytorch-cifar-models_master uv run python app.py --host 127.0.0.1 --port 8000
        servers=[{"url": "http://127.0.0.1:8000"}],

    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    ensure_privacy_registries()
    # Include routers
    app.include_router(api_router)
    app.state.config = config

    @app.get("/")
    def root():
        return {"message": "Welcome to the TITANN Job API"}

    return app


if __name__ == "__main__":
    uvicorn.run(
        "app:create_app",
        host=config.host,
        port=config.port,
        workers=config.workers,
        factory=True
    )
