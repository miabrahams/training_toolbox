"""Filesystem-backed loader for LoRA metadata dumps."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from .models import LoraRecord

from pydantic import ValidationError

LOGGER = logging.getLogger(__name__)


class LoraInfoError(RuntimeError):
    """Base error type for LoRA metadata issues."""


class LoraDirectoryNotFound(LoraInfoError):
    """Raised when the provided root directory does not exist."""


class LoraRecordNotFound(LoraInfoError):
    """Raised when a specific LoRA slug cannot be resolved."""


@dataclass(slots=True, frozen=True)
class LoraLoadError:
    """Captures failures that happened while parsing a directory."""

    slug: str
    path: Path
    reason: str


class LoraInfoClient:
    """Eagerly loads LoRA metadata from a flat directory structure."""

    def __init__(
        self,
        root_dir: Path,
        records: Sequence[LoraRecord],
        errors: Sequence[LoraLoadError] | None = None,
    ) -> None:
        self._root_dir = root_dir
        self._records = tuple(sorted(records, key=lambda r: r.slug.casefold()))
        self._by_slug = {record.slug.casefold(): record for record in self._records}
        self._tag_index = _build_index(self._records, lambda record: record.tags)
        self._trained_word_index = _build_index(
            self._records, lambda record: record.trained_words
        )
        self._errors = tuple(errors or ())

    @classmethod
    def from_directory(
        cls,
        root_dir: str | Path,
        *,
        encoding: str = "utf-8",
        strict: bool = False,
    ) -> "LoraInfoClient":
        path = Path(root_dir).expanduser()
        if not path.exists():
            raise LoraDirectoryNotFound(f"LoRA data directory '{path}' does not exist")
        if not path.is_dir():
            raise LoraDirectoryNotFound(f"LoRA data directory '{path}' is not a folder")

        records: list[LoraRecord] = []
        errors: list[LoraLoadError] = []
        for entry in sorted(path.iterdir()):
            if not entry.is_dir():
                continue
            info_path = entry / "info.json"
            slug = entry.name
            if not info_path.is_file():
                errors.append(
                    LoraLoadError(slug=slug, path=info_path, reason="missing info.json")
                )
                continue
            try:
                with info_path.open("r", encoding=encoding) as handle:
                    payload = json.load(handle)
            except (OSError, json.JSONDecodeError) as exc:
                errors.append(
                    LoraLoadError(slug=slug, path=info_path, reason=str(exc))
                )
                LOGGER.warning("Failed to read LoRA info at %s: %s", info_path, exc)
                continue
            try:
                record = LoraRecord.from_payload(slug=slug, directory=entry, payload=payload)
            except ValidationError as exc:
                errors.append(LoraLoadError(slug=slug, path=info_path, reason=str(exc)))
                LOGGER.warning("Failed to validate LoRA info at %s: %s", info_path, exc)
                continue
            except Exception as exc:  # pragma: no cover - defensive safety net
                errors.append(LoraLoadError(slug=slug, path=info_path, reason=str(exc)))
                LOGGER.warning("Failed to parse LoRA info at %s: %s", info_path, exc)
                continue
            records.append(record)

        if strict and errors:
            msgs = "; ".join(f"{error.slug}: {error.reason}" for error in errors)
            raise LoraInfoError(f"Encountered errors while loading LoRA data: {msgs}")

        return cls(root_dir=path, records=records, errors=errors)

    @property
    def root_dir(self) -> Path:
        return self._root_dir

    @property
    def load_errors(self) -> tuple[LoraLoadError, ...]:
        return self._errors

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._records)

    def __iter__(self):  # pragma: no cover - trivial
        return iter(self._records)

    def all(self) -> tuple[LoraRecord, ...]:
        return self._records

    def get(self, slug: str) -> LoraRecord | None:
        if not slug:
            return None
        return self._by_slug.get(slug.casefold())

    def require(self, slug: str) -> LoraRecord:
        record = self.get(slug)
        if record is None:
            raise LoraRecordNotFound(f"LoRA '{slug}' not found in {self._root_dir}")
        return record

    def find_by_tag(self, tag: str) -> list[LoraRecord]:
        if not tag:
            return []
        return list(self._tag_index.get(tag.casefold(), ()))

    def find_by_trained_word(self, word: str) -> list[LoraRecord]:
        if not word:
            return []
        return list(self._trained_word_index.get(word.casefold(), ()))

    def search(self, query: str, *, limit: int | None = None) -> list[LoraRecord]:
        if not query:
            return []
        needle = query.casefold()
        matches: list[LoraRecord] = []
        for record in self._records:
            if _record_matches(record, needle):
                matches.append(record)
                if limit is not None and len(matches) >= limit:
                    break
        return matches


def _build_index(
    records: Sequence[LoraRecord], extractor: Callable[[LoraRecord], Sequence[str]]
) -> dict[str, tuple[LoraRecord, ...]]:
    index: dict[str, list[LoraRecord]] = {}
    for record in records:
        seen: set[str] = set()
        for value in extractor(record):
            key = value.casefold()
            if key in seen:
                continue
            seen.add(key)
            index.setdefault(key, []).append(record)
    return {key: tuple(items) for key, items in index.items()}


def _record_matches(record: LoraRecord, needle: str) -> bool:
    fields: list[str] = [record.slug.casefold(), record.name.casefold()]
    if record.base_model:
        fields.append(record.base_model.casefold())
    if record.version_name:
        fields.append(record.version_name.casefold())
    for tag in record.tags:
        fields.append(tag.casefold())
    for word in record.trained_words:
        fields.append(word.casefold())
    return any(needle in field for field in fields)