# API Router for all endpoints
from fastapi import APIRouter

from attack_server.routers.info_router import router as info_router
# Import and include all sub-routers here
# from .dataset_router import router as dataset_router
from attack_server.routers.job_router import router as job_router
from attack_server.routers.model_router import router as model_router
from attack_server.routers.report_router import router as report_router
from attack_server.routers.repository_router import router as repository_router
from attack_server.routers.test_router import router as test_router

# from .privacy_router import router as privacy_router

api_router = APIRouter()

# api_router.include_router(dataset_router)
api_router.include_router(model_router)
api_router.include_router(job_router)
api_router.include_router(info_router)
api_router.include_router(report_router)
api_router.include_router(repository_router)
api_router.include_router(test_router)
# api_router.include_router(privacy_router)
