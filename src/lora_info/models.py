"""Data structures that describe parsed LoRA metadata."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

DATE_FORMATS = (
    "%Y-%m-%dT%H:%M:%S.%f%z",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d",
)


@dataclass(slots=True, frozen=True)
class LoraMediaAsset:
    """Single preview image or video entry attached to a LoRA."""

    url: str
    type: str | None = None
    nsfw_level: int | None = None
    has_metadata: bool | None = None
    prompt: str | None = None
    blurhash: str | None = None


@dataclass(slots=True, frozen=True)
class LoraRecord:
    """High-level metadata about a LoRA variant."""

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
    metadata: Mapping[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_payload(cls, slug: str, directory: Path, payload: Mapping[str, Any]) -> "LoraRecord":
        tags = _string_sequence(payload.get("tags"))
        trained_words = _string_sequence(payload.get("trained_words"))
        media = tuple(_parse_media(payload.get("images", ())))
        created_at = _parse_datetime(payload.get("createdDate"))
        updated_at = _parse_datetime(payload.get("updatedDate"))
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
        metadata = {k: payload[k] for k in set(payload.keys()) - known_keys}

        return cls(
            slug=slug,
            directory=directory,
            name=str(payload.get("name", slug)),
            model_id=_to_int(payload.get("modelId")),
            version_id=_to_int(payload.get("versionId")),
            version_name=_optional_str(payload.get("versionName")),
            kind=_optional_str(payload.get("type")),
            tags=tags,
            trained_words=trained_words,
            base_model=_optional_str(payload.get("baseModel")),
            path=_optional_path(payload.get("path")),
            subdir=_optional_str(payload.get("subdir")),
            nsfw=_to_bool(payload.get("nsfw")),
            nsfw_level=_to_int(payload.get("nsfwLevel")),
            created_at=created_at,
            updated_at=updated_at,
            version_description=_optional_str(payload.get("version_desc")),
            model_description=_optional_str(payload.get("model_desc")),
            media=media,
            metadata=metadata,
        )

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
            "media": [media.__dict__ for media in self.media],
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


def _parse_media(items: Iterable[Mapping[str, Any]]) -> Iterable[LoraMediaAsset]:
    for item in items or ():
        url = item.get("url")
        if not url:
            continue
        yield LoraMediaAsset(
            url=str(url),
            type=_optional_str(item.get("type")),
            nsfw_level=_to_int(item.get("nsfwLevel")),
            has_metadata=_to_bool(item.get("hasMeta")),
            prompt=_optional_str(item.get("prompt")),
            blurhash=_optional_str(item.get("blurhash")),
        )


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
