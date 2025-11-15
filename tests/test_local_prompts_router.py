from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

from fastapi import FastAPI
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.v1.routers.local_prompts import local_prompts_router
from src.controllers.prompts.processor import PromptProcessor
from src.db.prompt_database import PromptDatabase, Prompt, PromptFields


def _build_app(db: PromptDatabase) -> TestClient:
    app = FastAPI()
    app.state.prompt_processor = PromptProcessor(db)
    app.include_router(local_prompts_router, prefix="/api/v1")
    return TestClient(app)


def _seed(db: PromptDatabase) -> None:
    with db.SessionLocal.begin() as session:
        session.add(Prompt(file_path="/a.png", prompt="{}"))
        session.add(Prompt(file_path="/b.png", prompt="{}"))
        session.flush()
        session.add_all(
            [
                PromptFields(
                    file_path="/a.png",
                    name="hero",
                    positive_prompt="Hero prompt",
                    checkpoint="HeroXL",
                    processed=1,
                    last_updated=datetime(2024, 1, 4, 0, 0, 0),
                ),
                PromptFields(
                    file_path="/b.png",
                    name="villain",
                    positive_prompt="Villain prompt",
                    checkpoint="VillainXL",
                    processed=1,
                    last_updated=datetime(2024, 1, 5, 0, 0, 0),
                ),
            ]
        )


def test_list_endpoint(tmp_path):
    db = PromptDatabase(tmp_path / "api_prompts.sqlite")
    db.ensure_schema()
    _seed(db)

    with _build_app(db) as client:
        resp = client.get("/api/v1/local-prompts", params={"limit": 1})
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["total"] == 2
        assert payload["count"] == 1
        assert payload["items"][0]["positive_prompt"]


def test_get_and_stats(tmp_path):
    db = PromptDatabase(tmp_path / "api_prompts.sqlite")
    db.ensure_schema()
    _seed(db)

    with _build_app(db) as client:
        resp = client.get("/api/v1/local-prompts")
        first_id = resp.json()["items"][0]["id"]

        detail = client.get(f"/api/v1/local-prompts/{first_id}")
        assert detail.status_code == 200
        assert detail.json()["id"] == first_id

        stats = client.get("/api/v1/local-prompts/_stats")
        assert stats.status_code == 200
        body = stats.json()
        assert body["total"] == 2
        assert body["top_checkpoints"]

        random_resp = client.get("/api/v1/local-prompts/random", params={"limit": 1})
        assert random_resp.status_code == 200
        assert len(random_resp.json()) == 1
