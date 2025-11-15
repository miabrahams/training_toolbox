from fastapi import FastAPI
from contextlib import asynccontextmanager
from time import time
from pathlib import Path

from src.lib.config import load_settings
from src.lora_info import LoraInfoClient
from src.db.prompt_database import PromptDatabase
from src.controllers.prompts.processor import PromptProcessor

from src.api.v1.routers import lora_router, extract_router, local_prompts_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Application is starting up...")
    app.state.started_at = time()
    settings = load_settings()
    app.state.settings = settings

    lora_directory = settings['lora_sidebar.data']
    app.state.lora_client = LoraInfoClient.from_directory(lora_directory)

    db_path = Path(settings.get("prompt_db.path", "data/prompts.sqlite"))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_db = PromptDatabase(db_path)
    prompt_db.ensure_schema()
    app.state.prompt_db = prompt_db
    app.state.prompt_processor = PromptProcessor(prompt_db)
    try:
        yield
    finally:
        # Shutdown: clean up resources here
        pass

app = FastAPI(lifespan=lifespan)


@app.get("/")
def read_root():
    return {"Hello": "World"}


app.include_router(lora_router, prefix="/api/v1")
app.include_router(extract_router, prefix="/api/v1")
app.include_router(local_prompts_router, prefix="/api/v1")
