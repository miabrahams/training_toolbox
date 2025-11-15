from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.v1.routers.imageboard_router import imageboard_router
from src.db.e6sql import ImageboardDatabase, Post, Tag, PostTag


def _create_schema(db: ImageboardDatabase) -> None:
    statements = [
        """
        CREATE TABLE IF NOT EXISTS tags (
            tag_id BIGINT PRIMARY KEY,
            name VARCHAR NOT NULL,
            category INTEGER NOT NULL,
            count BIGINT NOT NULL,
            avg_score DOUBLE NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS posts (
            id BIGINT PRIMARY KEY,
            seq_id BIGINT NOT NULL,
            created_at TIMESTAMP NOT NULL,
            rating VARCHAR NOT NULL,
            image_width INTEGER NOT NULL,
            image_height INTEGER NOT NULL,
            fav_count INTEGER NOT NULL,
            file_ext VARCHAR NOT NULL,
            is_deleted BOOLEAN NOT NULL,
            score INTEGER NOT NULL,
            up_score INTEGER NOT NULL,
            down_score INTEGER NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS post_tags (
            post_id BIGINT NOT NULL,
            tag_id BIGINT NOT NULL
        )
        """,
    ]
    with db.engine.begin() as conn:
        for stmt in statements:
            conn.execute(text(stmt))


def _seed(db: ImageboardDatabase) -> None:
    _create_schema(db)
    with db.SessionLocal.begin() as session:
        tags = [
            Tag(tag_id=1, name="hero", category=0, count=10, avg_score=3.5),
            Tag(tag_id=2, name="villain", category=0, count=5, avg_score=2.0),
            Tag(tag_id=3, name="scenery", category=0, count=7, avg_score=4.0),
        ]
        session.add_all(tags)

        posts = [
            Post(
                id=100,
                seq_id=1,
                created_at=datetime(2024, 1, 1, 12, 0, 0),
                rating="g",
                image_width=1024,
                image_height=768,
                fav_count=50,
                file_ext="png",
                is_deleted=False,
                score=200,
                up_score=220,
                down_score=20,
            ),
            Post(
                id=101,
                seq_id=2,
                created_at=datetime(2024, 1, 2, 12, 0, 0),
                rating="q",
                image_width=800,
                image_height=600,
                fav_count=20,
                file_ext="jpg",
                is_deleted=False,
                score=120,
                up_score=140,
                down_score=20,
            ),
        ]
        session.add_all(posts)
        session.flush()

        session.add_all(
            [
                PostTag(post_id=100, tag_id=1),
                PostTag(post_id=100, tag_id=3),
                PostTag(post_id=101, tag_id=2),
            ]
        )


def _build_app(db: ImageboardDatabase) -> TestClient:
    app = FastAPI()
    app.state.imageboard_db = db
    app.include_router(imageboard_router, prefix="/api/v1")
    return TestClient(app)


def test_db_list_posts_and_tags(tmp_path):
    db = ImageboardDatabase(tmp_path / "e6.duckdb")
    _seed(db)

    posts, total = db.list_posts(include_tags=["hero"])
    assert total == 1
    assert posts[0].id == 100

    posts, total = db.list_posts(exclude_tags=["villain"])
    assert total == 1

    tags, total_tags = db.search_tags(search="hero")
    assert total_tags == 1
    assert tags[0].name == "hero"


def test_router_endpoints(tmp_path):
    db = ImageboardDatabase(tmp_path / "e6.duckdb")
    _seed(db)

    with _build_app(db) as client:
        resp = client.get("/api/v1/imageboard/posts", params={"tags": ["hero"]})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["items"][0]["id"] == 100

        detail = client.get("/api/v1/imageboard/posts/100")
        assert detail.status_code == 200
        assert detail.json()["tags"] == ["hero", "scenery"]

        tag_resp = client.get("/api/v1/imageboard/tags", params={"q": "vill"})
        assert tag_resp.status_code == 200
        assert tag_resp.json()["items"][0]["name"] == "villain"

        random_resp = client.get("/api/v1/imageboard/posts/random", params={"limit": 1})
        assert random_resp.status_code == 200
        assert len(random_resp.json()) == 1
