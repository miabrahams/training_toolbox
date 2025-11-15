from __future__ import annotations

from enum import Enum

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from src.controllers.prompts.processor import PromptProcessor
from src.api.v1.schemas import (
    LocalPrompt,
    LocalPromptListResponse,
    LocalPromptStats,
)


class PromptSort(str, Enum):
    updated = "updated"
    random = "random"
    steps = "steps"
    cfg = "cfg"
    width = "width"
    height = "height"
    seed = "seed"
    checkpoint = "checkpoint"


class SortDirection(str, Enum):
    asc = "asc"
    desc = "desc"


local_prompts_router = APIRouter(prefix="/local-prompts", tags=["local-prompts"])


def get_prompt_controller(request: Request) -> PromptProcessor:
    controller = getattr(request.app.state, "prompt_processor", None)
    if controller is None:
        raise HTTPException(status_code=500, detail="Prompt controller not initialized")
    return controller


def _serialize(records) -> list[LocalPrompt]:
    return [LocalPrompt.model_validate(record) for record in records]


@local_prompts_router.get("/", response_model=LocalPromptListResponse)
def list_prompts(
    q: str | None = Query(None, description="Search text in positive prompt"),
    checkpoint: str | None = Query(None, description="Filter by checkpoint substring"),
    lora: str | None = Query(None, description="Filter by LoRA substring"),
    has_lora: bool | None = Query(None, description="Require LoRA references"),
    has_ip: bool | None = Query(None, description="Require IP Adapter image"),
    ip_enabled: bool | None = Query(None, description="Filter on IP adapter enabled flag"),
    processed: bool | None = Query(None, description="Filter on processed flag"),
    min_steps: int | None = Query(None, ge=0),
    max_steps: int | None = Query(None, ge=0),
    min_cfg: float | None = Query(None, ge=0),
    max_cfg: float | None = Query(None, ge=0),
    width: int | None = Query(None, ge=0),
    height: int | None = Query(None, ge=0),
    aspect_ratio: str | None = Query(None),
    file_fragment: str | None = Query(None, description="Filter by file path fragment"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    sort_by: PromptSort = Query(PromptSort.updated),
    direction: SortDirection = Query(SortDirection.desc),
    controller: PromptProcessor = Depends(get_prompt_controller),
) -> LocalPromptListResponse:
    records, total = controller.query_prompts(
        search=q,
        checkpoint=checkpoint,
        lora=lora,
        has_lora=has_lora,
        has_ip_adapter=has_ip,
        ip_enabled=ip_enabled,
        processed=processed,
        min_steps=min_steps,
        max_steps=max_steps,
        min_cfg=min_cfg,
        max_cfg=max_cfg,
        width=width,
        height=height,
        aspect_ratio=aspect_ratio,
        file_fragment=file_fragment,
        limit=limit,
        offset=offset,
        sort_by=sort_by.value,
        sort_desc=direction == SortDirection.desc,
    )
    items = _serialize(records)
    return LocalPromptListResponse(
        total=total,
        count=len(items),
        offset=offset,
        limit=limit,
        items=items,
    )


@local_prompts_router.get("/random", response_model=list[LocalPrompt])
def random_prompts(
    limit: int = Query(1, ge=1, le=50),
    checkpoint: str | None = Query(None, description="Restrict to checkpoint substring"),
    controller: PromptProcessor = Depends(get_prompt_controller),
) -> list[LocalPrompt]:
    records = controller.random_prompts(limit=limit, checkpoint=checkpoint)
    return _serialize(records)


@local_prompts_router.get("/_stats", response_model=LocalPromptStats)
def prompt_stats(
    controller: PromptProcessor = Depends(get_prompt_controller),
) -> LocalPromptStats:
    return LocalPromptStats(**controller.stats())


@local_prompts_router.get("/{prompt_id}", response_model=LocalPrompt)
def get_prompt(
    prompt_id: int,
    controller: PromptProcessor = Depends(get_prompt_controller),
) -> LocalPrompt:
    record = controller.get_prompt(prompt_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Prompt {prompt_id} not found")
    return LocalPrompt.model_validate(record)
