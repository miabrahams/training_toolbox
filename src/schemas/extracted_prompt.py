from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict, field_validator

def _to_bool(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "on"}:
        return True
    if s in {"0", "false", "no", "off"}:
        return False
    return None

def _to_int(v):
    try:
        return int(v) if v is not None and str(v) != "" else None
    except Exception:
        return None

def _to_float(v):
    try:
        return float(v) if v is not None and str(v) != "" else None
    except Exception:
        return None

class ExtractedPrompt(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    # Schema bookkeeping
    schema_version: Optional[str] = None
    schema_name: Optional[str] = None

    # Core
    positive_prompt: str = Field(..., alias="positive_prompt")
    negative_prompt: Optional[str] = None
    cleaned_prompt: Optional[str] = None
    checkpoint: Optional[str] = None

    # Sampler
    steps: Optional[int] = None
    cfg: Optional[float] = None
    sampler_name: Optional[str] = None
    scheduler: Optional[str] = None
    seed: Optional[int] = None

    # Aspect/size
    aspect_ratio: Optional[str] = None
    swap_dimensions: Optional[bool] = None

    # LoRAs
    loras: Optional[str] = None

    # IP Adapter
    ip_enabled: Optional[bool] = None
    ip_image: Optional[str] = None
    ip_weight: Optional[float] = None

    # Toggles
    rescale_cfg: Optional[bool] = None
    perp_neg: Optional[bool] = None

    @field_validator("steps", "seed", mode="before")
    @classmethod
    def _ints(cls, v):
        return _to_int(v)

    @field_validator("cfg", "ip_weight", mode="before")
    @classmethod
    def _floats(cls, v):
        return _to_float(v)

    @field_validator("ip_enabled", "rescale_cfg", "perp_neg", "swap_dimensions", mode="before")
    @classmethod
    def _bools(cls, v):
        return _to_bool(v)