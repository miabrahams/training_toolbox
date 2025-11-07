from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)


class Resolution(BaseModel):
    model_config = ConfigDict(extra="ignore")
    width: int
    height: int

class Sampler(BaseModel):
    model_config = ConfigDict(extra="ignore")
    steps: Optional[int] = None
    cfg: Optional[float] = None
    sampler_name: Optional[str] = None
    scheduler: Optional[str] = None

class IPAdapter(BaseModel):
    model_config = ConfigDict(extra="ignore")
    image: Optional[str] = None  # path
    weight: Optional[float] = None  # 0-1
    enabled: Optional[bool] = None


class PromptReplaceDetail(BaseModel):
    model_config = ConfigDict(extra="ignore")
    positive_prompt: Optional[str] = None
    negative_prompt: Optional[str] = None
    resolution: Optional[Resolution] = None
    loras: Optional[str] = None
    sampler: Optional[Sampler] = None
    name: Optional[str] = None
    rescaleCfg: Optional[bool] = None
    perpNeg: Optional[bool] = None
    ipAdapter: Optional[IPAdapter] = None

    def to_wire(self) -> Dict[str, Any]:
        """
        Convert to a dict matching the frontend's expected shape,
        excluding any keys that are None.
        """
        return self.model_dump(exclude_none=True)