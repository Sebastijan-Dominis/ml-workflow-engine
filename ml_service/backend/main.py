import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import pipelines

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[str(os.getenv("FRONTEND_URL"))],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(pipelines.router)

@app.get("/")
async def health_check():
    return {"Healthy": 200}
