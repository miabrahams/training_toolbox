from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from src.lora_info import (
    LoraDirectoryNotFound,
    LoraInfoClient,
    LoraInfoError,
)


def test_client_loads_records_and_indexes(tmp_path: Path) -> None:
    root = tmp_path / "loras"
    root.mkdir()
    _write_info(
        root,
        slug="HeroBoss",
        name="Hero Boss",
        tags=["boss", "fantasy"],
        trained_words=["heroboss"],
        baseModel="SDXL",
        versionName="v1",
        images=[{"url": "https://example.com/a.png", "type": "image"}],
    )
    _write_info(
        root,
        slug="SciFiVillain",
        name="Sci-Fi Villain",
        tags=["scifi", "villain"],
        trained_words=[" cyber ", "villain"],
        baseModel="SDXL",
    )

    client = LoraInfoClient.from_directory(root)

    assert len(client.all()) == 2
    hero = client.require("HeroBoss")
    assert hero.name == "Hero Boss"
    assert hero.tags == ("boss", "fantasy")
    assert hero.trained_words == ("heroboss",)
    assert hero.media[0].url == "https://example.com/a.png"
    assert hero.created_at is not None
    assert hero.created_at.year == 2024
    assert hero.path is not None
    assert hero.path.name == "HeroBoss.safetensors"

    assert {record.slug for record in client.find_by_tag("boss")} == {"HeroBoss"}
    assert {record.slug for record in client.find_by_trained_word("villain")} == {"SciFiVillain"}
    assert {record.slug for record in client.search("hero")} == {"HeroBoss"}


def test_missing_directories_and_strict_mode(tmp_path: Path) -> None:
    root = tmp_path / "loras"
    root.mkdir()
    (root / "Broken").mkdir()
    _write_info(root, slug="ValidOne")

    client = LoraInfoClient.from_directory(root)
    assert client.load_errors  # missing info.json recorded

    with pytest.raises(LoraInfoError):
        LoraInfoClient.from_directory(root, strict=True)

    with pytest.raises(LoraDirectoryNotFound):
        LoraInfoClient.from_directory(tmp_path / "missing")


def _write_info(root: Path, slug: str, **overrides) -> None:
    directory = root / slug
    directory.mkdir(parents=True, exist_ok=True)
    data = {
        "name": slug,
        "modelId": 1,
        "versionId": 1,
        "versionName": "v1",
        "tags": ["tag"],
        "trained_words": ["trigger"],
        "baseModel": "SDXL",
        "images": [],
        "nsfw": False,
        "nsfwLevel": 1,
        "version_desc": None,
        "model_desc": None,
        "type": "LoRA",
        "createdDate": "2024-01-01T00:00:00Z",
        "updatedDate": "2024-01-02T00:00:00Z",
        "subdir": "",
        "path": f"/models/{slug}.safetensors",
    }
    data.update(overrides)
    info_path = directory / "info.json"
    info_path.write_text(json.dumps(data), encoding="utf-8")
