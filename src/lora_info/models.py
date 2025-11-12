"""Data structures that describe parsed LoRA metadata."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

DATE_FORMATS = (
    "%Y-%m-%dT%H:%M:%S.%f%z",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d",
)


class LoraMediaAsset(BaseModel):
    """Single preview image or video entry attached to a LoRA."""

    model_config = ConfigDict(frozen=True, populate_by_name=True, extra="ignore")

    url: str
    type: str | None = None
    nsfw_level: int | None = Field(default=None, alias="nsfwLevel")
    has_metadata: bool | None = Field(default=None, alias="hasMeta")
    prompt: str | None = None
    blurhash: str | None = None

    @field_validator("url", mode="before")
    @classmethod
    def _validate_url(cls, value: Any) -> str:
        text = _optional_str(value)
        if not text:
            raise ValueError("media asset requires a url")
        return text

    @field_validator("type", "prompt", "blurhash", mode="before")
    @classmethod
    def _normalize_optional_text(cls, value: Any) -> str | None:
        return _optional_str(value)

    @field_validator("nsfw_level", mode="before")
    @classmethod
    def _normalize_int(cls, value: Any) -> int | None:
        return _to_int(value)

    @field_validator("has_metadata", mode="before")
    @classmethod
    def _normalize_bool(cls, value: Any) -> bool | None:
        return _to_bool(value)


class LoraRecord(BaseModel):
    """High-level metadata about a LoRA variant."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    slug: str
    directory: Path
    name: str
    model_id: int | None = None
    version_id: int | None = None
    version_name: str | None = None
    kind: str | None = None
    tags: tuple[str, ...] = ()
    trained_words: tuple[str, ...] = ()
    base_model: str | None = None
    path: Path | None = None
    subdir: str | None = None
    nsfw: bool | None = None
    nsfw_level: int | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    version_description: str | None = None
    model_description: str | None = None
    media: tuple[LoraMediaAsset, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict, repr=False)

    @classmethod
    def from_payload(cls, slug: str, directory: Path, payload: Mapping[str, Any]) -> "LoraRecord":
        known_keys = {
            "name",
            "modelId",
            "versionId",
            "versionName",
            "tags",
            "trained_words",
            "baseModel",
            "images",
            "nsfw",
            "nsfwLevel",
            "version_desc",
            "model_desc",
            "type",
            "createdDate",
            "updatedDate",
            "subdir",
            "path",
        }
        metadata = {k: payload[k] for k in payload.keys() - known_keys}
        data: dict[str, Any] = {
            "slug": slug,
            "directory": directory,
            "name": payload.get("name", slug),
            "model_id": payload.get("modelId"),
            "version_id": payload.get("versionId"),
            "version_name": payload.get("versionName"),
            "kind": payload.get("type"),
            "tags": payload.get("tags"),
            "trained_words": payload.get("trained_words"),
            "base_model": payload.get("baseModel"),
            "path": payload.get("path"),
            "subdir": payload.get("subdir"),
            "nsfw": payload.get("nsfw"),
            "nsfw_level": payload.get("nsfwLevel"),
            "created_at": payload.get("createdDate"),
            "updated_at": payload.get("updatedDate"),
            "version_description": payload.get("version_desc"),
            "model_description": payload.get("model_desc"),
            "media": payload.get("images"),
            "metadata": metadata,
        }
        return cls.model_validate(data)

    @field_validator("name", "version_name", "kind", "base_model", "subdir", "version_description", "model_description", mode="before")
    @classmethod
    def _normalize_text(cls, value: Any) -> str | None:
        return _optional_str(value)

    @field_validator("tags", "trained_words", mode="before")
    @classmethod
    def _normalize_string_sequence(cls, value: Any) -> tuple[str, ...]:
        return _string_sequence(value)

    @field_validator("model_id", "version_id", "nsfw_level", mode="before")
    @classmethod
    def _normalize_optional_int(cls, value: Any) -> int | None:
        return _to_int(value)

    @field_validator("nsfw", mode="before")
    @classmethod
    def _normalize_optional_bool(cls, value: Any) -> bool | None:
        return _to_bool(value)

    @field_validator("created_at", "updated_at", mode="before")
    @classmethod
    def _normalize_datetime(cls, value: Any) -> datetime | None:
        return _parse_datetime(value)

    @field_validator("path", mode="before")
    @classmethod
    def _normalize_path(cls, value: Any) -> Path | None:
        return _optional_path(value)

    @field_validator("media", mode="before")
    @classmethod
    def _normalize_media(cls, value: Any) -> tuple[LoraMediaAsset, ...]:
        return _parse_media(value)

    @field_validator("metadata", mode="before")
    @classmethod
    def _normalize_metadata(cls, value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, Mapping):
            return dict(value)
        raise TypeError("metadata must be a mapping")

    def as_dict(self) -> dict[str, Any]:
        return {
            "slug": self.slug,
            "directory": str(self.directory),
            "name": self.name,
            "model_id": self.model_id,
            "version_id": self.version_id,
            "version_name": self.version_name,
            "kind": self.kind,
            "tags": list(self.tags),
            "trained_words": list(self.trained_words),
            "base_model": self.base_model,
            "path": str(self.path) if self.path else None,
            "subdir": self.subdir,
            "nsfw": self.nsfw,
            "nsfw_level": self.nsfw_level,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "version_description": self.version_description,
            "model_description": self.model_description,
            "media": [asset.model_dump(mode="python") for asset in self.media],
            "metadata": dict(self.metadata),
        }




def _string_sequence(values: Any) -> tuple[str, ...]:
    if not values:
        return ()
    if isinstance(values, str):
        return (values.strip(),) if values.strip() else ()
    result: list[str] = []
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            result.append(text)
    return tuple(result)


def _parse_media(items: Any) -> tuple[LoraMediaAsset, ...]:
    if not items:
        return ()
    result: list[LoraMediaAsset] = []
    for item in items:
        if isinstance(item, LoraMediaAsset):
            result.append(item)
            continue
        if not isinstance(item, Mapping):
            continue
        data = dict(item)
        if not data.get("url"):
            continue
        try:
            result.append(LoraMediaAsset.model_validate(data))
        except ValidationError:
            continue
    return tuple(result)



def _parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value)
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        pass
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _to_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def _optional_path(value: Any) -> Path | None:
    text = _optional_str(value)
    if not text:
        return None
    return Path(text)


__all__ = ["LoraMediaAsset", "LoraRecord"]
