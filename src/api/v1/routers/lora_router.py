import asyncio
from pathlib import Path
from typing import List, Dict, Any

from fastapi import APIRouter, HTTPException, Query, Depends, Request
from src.lora_info import LoraInfoClient


lora_router = APIRouter(prefix="/lora", tags=["lora"])


def get_client(request: Request) -> LoraInfoClient:
    client = getattr(request.app.state, "lora_client", None)
    if client is None:
        raise HTTPException(status_code=500, detail="LoRA client not initialized")
    return client

def get_settings(request: Request) -> Any:
    settings = getattr(request.app.state, "settings", None)
    if settings is None:
        raise HTTPException(status_code=500, detail="Settings not initialized")
    return settings


def _serialize(record) -> dict[str, Any]:
    return record.as_dict()

@lora_router.get("/", summary="List LoRAs")
def list_loras(
    q: str | None = Query(None, description="Search text across slug/name/tags/words"),
    tag: str | None = Query(None, description="Filter by tag"),
    word: str | None = Query(None, description="Filter by trained word"),
    client: LoraInfoClient = Depends(get_client),
) -> List[Dict[str, Any]]:
    if q: records = client.search(q)
    elif tag: records = client.find_by_tag(tag)
    elif word: records = client.find_by_trained_word(word)
    else: records = list(client.all())
    return [r.as_dict() for r in records]


@lora_router.get("/{slug}", summary="Get a LoRA by slug")
def get_lora(
    slug: str,
    client: LoraInfoClient = Depends(get_client),
) -> dict[str, Any]:
    record = client.get(slug)
    if record is None:
        raise HTTPException(status_code=404, detail=f"LoRA '{slug}' not found")
    return _serialize(record)


@lora_router.get("/_meta/errors", summary="Loader errors (if any)")
def load_errors(
    client: LoraInfoClient = Depends(get_client),
) -> list[dict[str, Any]]:
    return [
        {"slug": e.slug, "path": str(e.path), "reason": e.reason}
        for e in client.load_errors
    ]


@lora_router.get("/_meta/stats", summary="Basic stats")
def stats(
    client: LoraInfoClient = Depends(get_client),
) -> dict[str, int]:
    records = client.all()
    tags = set()
    words = set()
    for r in records:
        tags.update(r.tags)
        words.update(r.trained_words)
    return {
        "total": len(records),
        "errors": len(client.load_errors),
        "unique_tags": len(tags),
        "unique_trained_words": len(words),
    }