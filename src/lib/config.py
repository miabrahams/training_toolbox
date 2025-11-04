"""Centralized configuration loading utilities.

This module exposes helpers around Dynaconf so every component (CLIs, UI,
notebooks, background scripts) can hydrate the same configuration hierarchy
from ``config/config.yml`` and ``config/secrets.yml``.  The loader supports:

* Automatic discovery of the config directory (defaulting to ``repo_root/config``
  but overridable via the ``TRAINING_TOOLBOX_CONFIG_DIR`` environment variable).
* Lazy, cached access so repeated lookups are inexpensive.
* Optional validation/normalisation through ``dynaconf.Validator`` objects.
* Simple helpers (`get_settings`, `get_section`, `get_path`) that work in both
  imperative scripts and notebooks without needing to instantiate classes.

Dynaconf keeps values mutable at runtime, enabling future expansion into
per-environment overrides or additional split configuration files without
requiring schema definitions ahead of time.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Sequence

from dynaconf import Dynaconf, Validator
from dynaconf.base import LazySettings

ENVVAR_PREFIX = "TRAINING_TOOLBOX"
CONFIG_DIR_ENVVAR = f"{ENVVAR_PREFIX}_CONFIG_DIR"
DEFAULT_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
# Order matters: later files override earlier ones.
BASE_CONFIG_FILES: Sequence[str] = ("config.yml", "secrets.yml")


def _resolve_config_dir(config_dir: Optional[Path | str] = None) -> Path:
    if config_dir:
        return Path(config_dir).expanduser().resolve()
    env_dir = os.getenv(CONFIG_DIR_ENVVAR)
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    return DEFAULT_CONFIG_DIR


def _iter_config_files(config_dir: Path) -> Iterator[Path]:
    for filename in BASE_CONFIG_FILES:
        path = config_dir / filename
        if path.exists():
            yield path
    # Allow optional drop-in overrides such as config.local.yml or files in
    # config.d/. They are completely optional but enable future splitting.
    local_file = config_dir / "config.local.yml"
    if local_file.exists():
        yield local_file
    config_dot_d = config_dir / "config.d"
    if config_dot_d.is_dir():
        for extra in sorted(config_dot_d.glob("*.yml")):
            yield extra


@lru_cache(maxsize=None)
def _load_settings(config_dir: str) -> LazySettings:
    config_path = Path(config_dir)
    files = [str(p) for p in _iter_config_files(config_path)]
    settings = Dynaconf(
        envvar_prefix=ENVVAR_PREFIX,
        settings_files=files,
        environments=True,
        load_dotenv=True,
        merge_enabled=True,
    )

    validators = [
        Validator("ui.server.port", cast=int, default=7000),
        Validator("ui.defaults.db_path", default="data/prompts.sqlite"),
        Validator("ui.defaults.data_dir", default="data"),
        Validator("captioner.input_dir", default="./data/input"),
        Validator("captioner.output_dir", default="./data/output"),
        Validator("captioner.error_dir", default="./data/errors"),
        Validator("captioner.collator.output_file", default="./data/collated_captions.txt"),
        Validator("captioner.collator.postprocess", cast=bool, default=True),
        Validator("captioner.embeddings.captions_file", default="captioner/data/collated_captions.txt"),
        Validator("captioner.embeddings.parquet", default="captioner/data/embeddings.parquet"),
        Validator("captioner.embeddings.model", default="text-embedding-3-large"),
        Validator("tag_versions.database", default="data/caption_versions/caption_versions.sqlite"),
        Validator("batch_tagger.model_path", default="models/eva02.pth"),
        Validator("batch_tagger.tags_path", default="models/tags.json"),
        Validator("duplicate_inspector.base_path", default="data"),
        Validator("duplicate_inspector.html.output_file", default="data/duplicates.html"),
        Validator("duplicate_inspector.html.path_rewrite.from", default=""),
        Validator("duplicate_inspector.html.path_rewrite.to", default=""),
        Validator("tools.check_db.default_path", default="data/prompts.sqlite"),
        Validator("tools.comfy.data_path", default="data"),
        Validator("tag_analyzer.analysis_filename", default="analysis_data.pkl"),
    ]
    settings.validators.register(*validators)
    settings.validators.validate()

    return settings


def get_settings(config_dir: Optional[Path | str] = None) -> LazySettings:
    """Return the cached Dynaconf settings instance."""
    resolved = _resolve_config_dir(config_dir)
    return _load_settings(str(resolved))


def reload_settings(config_dir: Optional[Path | str] = None) -> LazySettings:
    """Clear the cache and reload configuration from disk."""
    resolved = _resolve_config_dir(config_dir)
    _load_settings.cache_clear()
    return _load_settings(str(resolved))


def get_section(section: str, default: Any = None, *, config_dir: Optional[Path | str] = None) -> Any:
    """Convenience helper to fetch a namespaced section (e.g. "captioner")."""
    settings = get_settings(config_dir)
    return settings.get(section, default)


def get_path(path: str, default: Any = None, *, config_dir: Optional[Path | str] = None) -> Any:
    """Retrieve a dotted path (``captioner.embeddings.parquet``)."""
    settings = get_settings(config_dir)
    return settings.get(path, default)


def register_validators(validators: Iterable[Validator], *, config_dir: Optional[Path | str] = None) -> None:
    settings = get_settings(config_dir)
    settings.validators.register(*validators)
    settings.validators.validate()


# Backwards-compatible wrappers -------------------------------------------------

def load_config(config_dir: Optional[Path | str] = None) -> LazySettings:
    return get_settings(config_dir)


def load_secrets(config_dir: Optional[Path | str] = None) -> LazySettings:
    # Secrets live in the same Dynaconf instance.
    return get_settings(config_dir)


def load_ui_config(config_dir: Optional[Path | str] = None) -> LazySettings:
    return get_settings(config_dir)


__all__ = [
    "get_settings",
    "reload_settings",
    "get_section",
    "get_path",
    "register_validators",
    "load_config",
    "load_secrets",
    "load_ui_config",
]
