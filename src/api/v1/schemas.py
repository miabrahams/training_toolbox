from __future__ import annotations

from datetime import datetime
from typing import List

from pydantic import BaseModel, ConfigDict


class LocalPrompt(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    file_path: str
    name: str | None = None
    positive_prompt: str
    cleaned_prompt: str | None = None
    negative_prompt: str | None = None
    checkpoint: str | None = None
    width: int | None = None
    height: int | None = None
    aspect_ratio: str | None = None
    swap_dimensions: bool | None = None
    loras: str | None = None
    steps: int | None = None
    cfg: float | None = None
    sampler_name: str | None = None
    scheduler: str | None = None
    seed: int | None = None
    rescale_cfg: bool | None = None
    perp_neg: bool | None = None
    ip_image: str | None = None
    ip_weight: float | None = None
    ip_enabled: bool | None = None
    processed: int | None = None
    last_updated: datetime | None = None


class LocalPromptListResponse(BaseModel):
    total: int
    count: int
    offset: int
    limit: int
    items: List[LocalPrompt]


class CheckpointStat(BaseModel):
    name: str | None
    count: int


class LocalPromptStats(BaseModel):
    total: int
    processed: int
    with_lora: int
    with_ip_adapter: int
    with_ip_enabled: int
    unique_checkpoints: int
    latest_update: datetime | None
    avg_steps: float | None = None
    avg_cfg: float | None = None
    top_checkpoints: List[CheckpointStat]


class ImageboardPost(BaseModel):
    id: int
    seq_id: int
    created_at: datetime
    rating: str
    image_width: int
    image_height: int
    fav_count: int
    file_ext: str
    is_deleted: bool
    score: int
    up_score: int
    down_score: int
    tags: List[str]


class ImageboardPostListResponse(BaseModel):
    total: int
    count: int
    offset: int
    limit: int
    items: List[ImageboardPost]


class ImageboardTag(BaseModel):
    tag_id: int
    name: str
    category: int
    count: int
    avg_score: float


class ImageboardTagListResponse(BaseModel):
    total: int
    count: int
    offset: int
    limit: int
    items: List[ImageboardTag]
