from fastapi import FastAPI, Request, HTTPException
from contextlib import asynccontextmanager
from time import time
from typing import Any

from src.lib.config import load_settings
from src.lora_info import LoraInfoClient

from src.api.v1.routers.lora_router import lora_router


app = FastAPI()

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Application is starting up...")
    app.state.started_at = time()
    settings = load_settings()
    app.state.settings = settings

    lora_directory = settings['lora_sidebar.data']
    app.state.lora_client = LoraInfoClient.from_directory(lora_directory)
    try:
        yield
    finally:
        # Shutdown: clean up resources here
        pass

@app.get("/")
def read_root():
    return {"Hello": "World"}


app.include_router(lora_router, prefix="/api/v1")