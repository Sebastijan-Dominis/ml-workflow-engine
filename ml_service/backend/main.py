"""Main entry point for the ML service backend."""
import os

import dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded

from ml_service.backend.limiter import limiter
from ml_service.backend.routers.features import router as features_router
from ml_service.backend.routers.modeling import router as modeling_router
from ml_service.backend.routers.pipelines import router as pipelines_router

dotenv.load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

app = FastAPI()

app.state.limiter = limiter

app.add_middleware(
    CORSMiddleware,
    allow_origins=[str(os.getenv("FRONTEND_URL"))],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

routers = [pipelines_router, modeling_router, features_router]

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
