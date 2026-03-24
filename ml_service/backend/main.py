"""Main entry point for the ML service backend."""
import os

import dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded

from ml_service.backend.limiter import limiter
from ml_service.backend.routers.data import router as data_router
from ml_service.backend.routers.dir_viewer import router as dir_viewer_router
from ml_service.backend.routers.features import router as features_router
from ml_service.backend.routers.file_viewer import router as file_viewer_router
from ml_service.backend.routers.modeling import router as modeling_router
from ml_service.backend.routers.pipeline_cfg import router as pipeline_cfg_router
from ml_service.backend.routers.pipelines import router as pipelines_router
from ml_service.backend.routers.promotion_thresholds import router as promotion_thresholds_router
from ml_service.backend.routers.scripts import router as scripts_router

dotenv.load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

app = FastAPI()

app.state.limiter = limiter

app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("ML_SERVICE_FRONTEND_URL", "http://localhost:8050")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

routers = [
    pipelines_router,
    modeling_router,
    features_router,
    data_router,
    pipeline_cfg_router,
    promotion_thresholds_router,
    scripts_router,
    file_viewer_router,
    dir_viewer_router,
]

for router in routers:
    app.include_router(router)


async def rate_limit_exceeded_handler(request, exc):
    """Handle rate limit exceeded exceptions."""
    return JSONResponse(
        status_code=429,
        content={"message": "Rate limit exceeded. Please try again later."},
    )

app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)

@app.get("/")
async def health_check():
    return {"Healthy": 200}
