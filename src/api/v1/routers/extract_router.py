from typing import Any
from fastapi import APIRouter, UploadFile, File, HTTPException
from io import BytesIO

from src.lib.metadata import comfy_metadata, memory_image_info
from src.lib.comfy_schemas.extractor import extract_from_json
from src.lib.errors import ExtractionFailedError
from src.schemas.prompt import ImagePrompt

extract_router = APIRouter(prefix="/extract", tags=["extract"])


# TODO make the metadata extractor a dependency
@extract_router.post("/image", summary="Extract prompt from a PNG image")
async def extract_from_image(file: UploadFile = File(...)) -> ImagePrompt:
    if file.content_type not in ("image/png", "application/octet-stream"):
        raise HTTPException(status_code=415, detail="Only PNG images are supported")

    data = await file.read()
    image_info = memory_image_info(BytesIO(data))

    try:
        prompt_graph, _ = comfy_metadata(image_info)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read Comfy metadata: {e}")

    try:
        extracted = extract_from_json(prompt_graph)  # tries all schemas newest-first
    except ExtractionFailedError as e:
        raise HTTPException(status_code=422, detail=f"Schema extraction failed: {e}")

    # Downcast to the shared base type so we only expose the common prompt fields.
    return ImagePrompt.model_validate(extracted)
