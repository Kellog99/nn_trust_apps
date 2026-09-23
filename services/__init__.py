def __getattr__(name: str):
    if name != "api_router":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from fastapi import APIRouter

    from services.dataset_router import router as dataset_router
    from services.info_router import router as info_router
    from services.job_router import router as job_router
    from services.model_router import router as model_router
    from services.report_router import router as report_router
    from services.repository_router import router as repository_router
    from services.single_attack import router as single_attack

    router = APIRouter()
    router.include_router(dataset_router)
    router.include_router(model_router)
    router.include_router(job_router)
    router.include_router(info_router)
    router.include_router(report_router)
    router.include_router(repository_router)
    router.include_router(single_attack)
    globals()[name] = router
    return router
