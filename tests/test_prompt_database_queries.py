from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.db.prompt_database import PromptDatabase, Prompt, PromptFields


def _seed_prompts(db: PromptDatabase) -> None:
    with db.SessionLocal.begin() as session:
        prompts = [
            Prompt(file_path="/images/hero.png", prompt="{}"),
            Prompt(file_path="/images/villain.png", prompt="{}"),
            Prompt(file_path="/images/other.png", prompt="{}"),
        ]
        session.add_all(prompts)
        session.flush()
        session.add_all(
            [
                PromptFields(
                    file_path="/images/hero.png",
                    name="hero",
                    positive_prompt="A brave hero fights a dragon",
                    cleaned_prompt="brave hero fights dragon",
                    checkpoint="HeroXL",
                    steps=30,
                    cfg=5.5,
                    loras="hero_lora:1.0",
                    processed=1,
                    ip_image="ip1.png",
                    ip_enabled=True,
                    last_updated=datetime(2024, 1, 1, 12, 0, 0),
                ),
                PromptFields(
                    file_path="/images/villain.png",
                    name="villain",
                    positive_prompt="A villain in neon city",
                    checkpoint="VillainXL",
                    steps=20,
                    cfg=6.0,
                    loras=None,
                    processed=1,
                    ip_image=None,
                    last_updated=datetime(2024, 1, 2, 12, 0, 0),
                ),
                PromptFields(
                    file_path="/images/other.png",
                    name="other",
                    positive_prompt="Landscape with mountains",
                    checkpoint="HeroXL",
                    steps=15,
                    cfg=4.0,
                    loras="",
                    processed=0,
                    ip_image=None,
                    last_updated=datetime(2024, 1, 3, 12, 0, 0),
                ),
            ]
        )


def test_query_prompts_filters(tmp_path):
    db = PromptDatabase(tmp_path / "prompts.sqlite")
    db.ensure_schema()
    _seed_prompts(db)

    records, total = db.query_prompt_fields(search="hero")
    assert total == 1
    assert records[0].file_path == "/images/hero.png"

    records, total = db.query_prompt_fields(has_lora=False)
    assert total == 2  # one empty string counts as no LoRA

    records, _ = db.query_prompt_fields(has_ip_adapter=True)
    assert len(records) == 1
    assert records[0].name == "hero"

    records, _ = db.query_prompt_fields(processed=False)
    assert len(records) == 1
    assert records[0].name == "other"


def test_random_and_stats(tmp_path):
    db = PromptDatabase(tmp_path / "prompts.sqlite")
    db.ensure_schema()
    _seed_prompts(db)

    random_records = db.get_random_prompts(limit=2)
    assert len(random_records) == 2

    stats = db.prompt_stats()
    assert stats["total"] == 3
    assert stats["processed"] == 2
    assert stats["with_lora"] == 1
    assert stats["with_ip_adapter"] == 1
    assert stats["unique_checkpoints"] == 2
    assert len(stats["top_checkpoints"]) >= 1
