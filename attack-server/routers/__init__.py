# API Router for all endpoints
from fastapi import APIRouter
# Import and include all sub-routers here
from .dataset_router import router as dataset_router
from .job_router import router as job_router
from .model_router import router as model_router
from .attack_router import router as attack_router


api_router = APIRouter()

api_router.include_router(dataset_router)
api_router.include_router(model_router)
api_router.include_router(job_router)
api_router.include_router(attack_router)