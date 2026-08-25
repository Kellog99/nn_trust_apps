# API Router for all endpoints
from fastapi import APIRouter

# Import and include all sub-routers here
from services.dataset_router import router as dataset_router
from services.info_router import router as info_router
from services.job_router import router as job_router
from services.model_router import router as model_router
from services.report_router import router as report_router
from services.repository_router import router as repository_router
from services.test_router import router as test_router


api_router = APIRouter()

api_router.include_router(dataset_router)
api_router.include_router(model_router)
api_router.include_router(job_router)
api_router.include_router(info_router)
api_router.include_router(report_router)
api_router.include_router(repository_router)
api_router.include_router(test_router)
