from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import api_router
from .config import get_settings
from .utils import setup_logger

settings = get_settings()
logger = setup_logger("glaucoma_oct")

app = FastAPI(
    title="Glaucoma OCT AI Platform",
    description="眼科医学 AI 平台 - 青光眼辅助诊断系统\n\n"
                "基于 3D-ResNet 的视网膜神经纤维层（RNFL）分割与厚度分析系统",
    version=settings.app_version,
    debug=settings.debug,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix=settings.api_prefix)
app.include_router(api_router, prefix="")


@app.middleware("http")
async def add_process_time_header(request, call_next):
    import time
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = (time.perf_counter() - start_time) * 1000
    response.headers["X-Process-Time-MS"] = f"{process_time:.2f}"
    return response


if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting server on {settings.host}:{settings.port}")
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        workers=1
    )
