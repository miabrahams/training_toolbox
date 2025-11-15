from __future__ import annotations

from enum import Enum
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from src.api.v1.schemas import (
    ImageboardPost,
    ImageboardPostListResponse,
    ImageboardTag,
    ImageboardTagListResponse,
)
from src.db.e6sql import ImageboardDatabase, Categories, Post, Tag


class PostSort(str, Enum):
    created = "created"
    score = "score"
    favs = "favs"
    width = "width"
    height = "height"
    post_id = "id"
    random = "random"


class TagSort(str, Enum):
    tag_name = "tag_name"
    avg_score = "avg_score"


class SortDirection(str, Enum):
    asc = "asc"
    desc = "desc"


imageboard_router = APIRouter(prefix="/imageboard", tags=["imageboard"])


def get_db(request: Request) -> ImageboardDatabase:
    db = getattr(request.app.state, "imageboard_db", None)
    if db is None:
        raise HTTPException(status_code=500, detail="Imageboard database not configured")
    return db


def _normalize_ratings(values: List[str] | None) -> List[str] | None:
    if not values:
        return None
    normalized = []
    allowed = {"g", "q", "e"}
    for value in values:
        if value is None:
            continue
        val = value.lower()
        if val not in allowed:
            raise HTTPException(status_code=400, detail=f"Invalid rating '{value}'")
        normalized.append(val)
    return normalized or None


def _category_to_id(value: str | None) -> int | None:
    if value is None:
        return None
    value = value.strip().lower()
    if not value:
        return None
    if value.isdigit():
        return int(value)
    for ident, label in Categories.items():
        if label == value:
            return ident
    raise HTTPException(status_code=400, detail=f"Unknown category '{value}'")


def _serialize_post(post: Post) -> ImageboardPost:
    tags = [pt.tag.name for pt in post.post_tags if pt.tag]
    return ImageboardPost(
        id=post.id,
        seq_id=post.seq_id,
        created_at=post.created_at,
        rating=post.rating,
        image_width=post.image_width,
        image_height=post.image_height,
        fav_count=post.fav_count,
        file_ext=post.file_ext,
        is_deleted=bool(post.is_deleted),
        score=post.score,
        up_score=post.up_score,
        down_score=post.down_score,
        tags=tags,
    )


def _serialize_tag(tag: Tag) -> ImageboardTag:
    return ImageboardTag(
        tag_id=int(tag.tag_id),
        name=tag.name,
        category=tag.category,
        count=int(tag.count),
        avg_score=float(tag.avg_score),
    )


@imageboard_router.get("/posts", response_model=ImageboardPostListResponse)
def list_posts(
    tags: List[str] | None = Query(None, description="Require posts to include all of these tags"),
    exclude: List[str] | None = Query(
        None, description="Filter out posts containing any of these tags"
    ),
    ratings: List[str] | None = Query(None, description="Rating codes (g/q/e)"),
    min_score: int | None = Query(None),
    min_favs: int | None = Query(None),
    min_width: int | None = Query(None),
    max_width: int | None = Query(None),
    min_height: int | None = Query(None),
    max_height: int | None = Query(None),
    file_ext: str | None = Query(None),
    include_deleted: bool = Query(False),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    sort_by: PostSort = Query(PostSort.created),
    direction: SortDirection = Query(SortDirection.desc),
    db: ImageboardDatabase = Depends(get_db),
) -> ImageboardPostListResponse:
    rating_filter = _normalize_ratings(ratings)
    random_order = sort_by is PostSort.random
    posts, total = db.list_posts(
        include_tags=tags,
        exclude_tags=exclude,
        ratings=rating_filter,
        min_score=min_score,
        min_favs=min_favs,
        min_width=min_width,
        max_width=max_width,
        min_height=min_height,
        max_height=max_height,
        file_ext=file_ext,
        include_deleted=include_deleted,
        limit=limit,
        offset=offset,
        sort_by=sort_by.value,
        descending=direction == SortDirection.desc,
        random_order=random_order,
    )
    items = [_serialize_post(post) for post in posts]
    return ImageboardPostListResponse(
        total=total,
        count=len(items),
        offset=offset,
        limit=limit,
        items=items,
    )


@imageboard_router.get("/posts/random", response_model=list[ImageboardPost])
def random_posts(
    limit: int = Query(1, ge=1, le=50),
    tags: List[str] | None = Query(None),
    ratings: List[str] | None = Query(None),
    min_score: int | None = Query(None),
    db: ImageboardDatabase = Depends(get_db),
) -> list[ImageboardPost]:
    rating_filter = _normalize_ratings(ratings)
    posts = db.random_posts(
        limit=limit,
        include_tags=tags,
        ratings=rating_filter,
        min_score=min_score,
    )
    return [_serialize_post(post) for post in posts]


@imageboard_router.get("/posts/{post_id}", response_model=ImageboardPost)
def get_post(post_id: int, db: ImageboardDatabase = Depends(get_db)) -> ImageboardPost:
    post = db.get_post(post_id)
    if post is None:
        raise HTTPException(status_code=404, detail="Post not found")
    return _serialize_post(post)


@imageboard_router.get("/tags", response_model=ImageboardTagListResponse)
def list_tags(
    q: str | None = Query(None, description="Search substring in tag names"),
    category: str | None = Query(None, description="Category id or name"),
    min_count: int | None = Query(None, ge=0),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    sort_by: TagSort = Query(TagSort.count),
    direction: SortDirection = Query(SortDirection.desc),
    db: ImageboardDatabase = Depends(get_db),
) -> ImageboardTagListResponse:
    category_id = _category_to_id(category)
    tags, total = db.search_tags(
        search=q,
        category=category_id,
        min_count=min_count,
        limit=limit,
        offset=offset,
        sort_by=sort_by.value,
        descending=direction == SortDirection.desc,
    )
    items = [_serialize_tag(tag) for tag in tags]
    return ImageboardTagListResponse(
        total=total,
        count=len(items),
        offset=offset,
        limit=limit,
        items=items,
    )


@imageboard_router.get("/tags/{name}", response_model=ImageboardTag)
def get_tag(name: str, db: ImageboardDatabase = Depends(get_db)) -> ImageboardTag:
    tag = db.get_tag(name)
    if tag is None:
        raise HTTPException(status_code=404, detail="Tag not found")
    return _serialize_tag(tag)
