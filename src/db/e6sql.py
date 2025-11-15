from datetime import datetime
from typing import List, Optional, Sequence, Tuple, Set

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Double,
    ForeignKey,
    Integer,
    String,
    create_engine,
    func,
    select,
)

import random

from sqlalchemy.orm import sessionmaker, relationship, Session, DeclarativeBase, Mapped, mapped_column, relationship, selectinload

class E6Base(DeclarativeBase):
    pass


Categories = {
	0: "general",     # most tags
	1: "artist",      # ok
	2: "contributor", # eg. modeler of an asset, not cg artist
	3: "copyright",   # eg "nintendo"
	4: "character",   # ok
	5: "species",     # ok
	6: "lore",        # rare
	7: "meta",        # ok
	8: "invalid",     # ok
}

Ratings = {
	"g": "general",
	"q": "questionable",
	"e": "explicit",
}



class Tag(E6Base):
    """Tags table with categorization."""

    __tablename__ = 'tags'

    tag_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False, unique=True, index=True)
    category: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    count: Mapped[int] = mapped_column(BigInteger, nullable=False)
    avg_score: Mapped[float] = mapped_column(Double, nullable=False)

    post_tags: Mapped[List["PostTag"]] = relationship(back_populates="tag")

    def __repr__(self):
        return f"<Tag(tag_id={self.tag_id}, name='{self.name}', category={self.category})>"


class Post(E6Base):
    """Posts table with metadata."""

    __tablename__ = "posts"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    seq_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    rating: Mapped[str] = mapped_column(String, nullable=False)
    image_width: Mapped[int] = mapped_column(Integer, nullable=False)
    image_height: Mapped[int] = mapped_column(Integer, nullable=False)
    fav_count: Mapped[int] = mapped_column(Integer, nullable=False)
    file_ext: Mapped[str] = mapped_column(String, nullable=False)
    is_deleted: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    score: Mapped[int] = mapped_column(Integer, nullable=False)
    up_score: Mapped[int] = mapped_column(Integer, nullable=False)
    down_score: Mapped[int] = mapped_column(Integer, nullable=False)

    # Relationships
    post_tags: Mapped[List["PostTag"]] = relationship(back_populates="post")

    def __repr__(self):
        return f"<Post(id={self.id}, rating='{self.rating}', score={self.score})>"


class PostTag(E6Base):
    """Junction table linking posts to tags."""

    __tablename__ = "post_tags"

    post_id: Mapped[BigInteger] = mapped_column(BigInteger, ForeignKey("posts.id"), primary_key=True)
    tag_id: Mapped[BigInteger] = mapped_column(BigInteger, ForeignKey("tags.tag_id"), primary_key=True)

    post: Mapped["Post"] = relationship("Post", back_populates="post_tags")
    tag: Mapped["Tag"] = relationship("Tag", back_populates="post_tags")

    def __repr__(self):
        return f"<PostTag(post_id={self.post_id}, tag_id={self.tag_id})>"


class ImageboardDatabase:
    """DuckDB-backed helper for browsing posts/tags metadata."""

    def __init__(self, db_path):
        self.engine = create_engine(f"duckdb:///{db_path}")
        self.SessionLocal = sessionmaker(bind=self.engine, class_=Session)

    def list_posts(
        self,
        *,
        include_tags: Optional[Sequence[str]] = None,
        exclude_tags: Optional[Sequence[str]] = None,
        ratings: Optional[Sequence[str]] = None,
        min_score: Optional[int] = None,
        min_favs: Optional[int] = None,
        min_width: Optional[int] = None,
        max_width: Optional[int] = None,
        min_height: Optional[int] = None,
        max_height: Optional[int] = None,
        file_ext: Optional[str] = None,
        include_deleted: bool = False,
        limit: int = 50,
        offset: int = 0,
        sort_by: str = "created",
        descending: bool = True,
        random_order: bool = False,
    ) -> Tuple[List[Post], int]:
        limit = max(1, min(limit, 200))
        offset = max(0, offset)

        with self.SessionLocal() as session:
            query = session.query(Post).options(
                selectinload(Post.post_tags).selectinload(PostTag.tag)
            )

            if include_tags:
                include_subq = (
                    session.query(PostTag.post_id)
                    .join(Tag, PostTag.tag_id == Tag.tag_id)
                    .filter(Tag.name.in_(include_tags))
                    .group_by(PostTag.post_id)
                    .having(func.count(func.distinct(Tag.name)) == len(include_tags))
                    .subquery()
                )
                query = query.filter(Post.id.in_(select(include_subq.c.post_id)))

            if exclude_tags:
                exclude_subq = (
                    session.query(PostTag.post_id)
                    .join(Tag, PostTag.tag_id == Tag.tag_id)
                    .filter(Tag.name.in_(exclude_tags))
                    .subquery()
                )
                query = query.filter(~Post.id.in_(select(exclude_subq.c.post_id)))

            if ratings:
                query = query.filter(Post.rating.in_(ratings))

            if min_score is not None:
                query = query.filter(Post.score >= min_score)
            if min_favs is not None:
                query = query.filter(Post.fav_count >= min_favs)
            if min_width is not None:
                query = query.filter(Post.image_width >= min_width)
            if max_width is not None:
                query = query.filter(Post.image_width <= max_width)
            if min_height is not None:
                query = query.filter(Post.image_height >= min_height)
            if max_height is not None:
                query = query.filter(Post.image_height <= max_height)
            if file_ext:
                query = query.filter(Post.file_ext == file_ext)
            if not include_deleted:
                query = query.filter(Post.is_deleted.is_(False))

            total = int(query.with_entities(func.count()).order_by(None).scalar() or 0)

            if random_order:
                query = query.order_by(func.random())
            else:
                sort_columns = {
                    "created": Post.created_at,
                    "score": Post.score,
                    "favs": Post.fav_count,
                    "width": Post.image_width,
                    "height": Post.image_height,
                    "id": Post.id,
                }
                column = sort_columns.get(sort_by, Post.created_at)
                direction = column.desc() if descending else column.asc()
                query = query.order_by(direction)

            posts = query.offset(offset).limit(limit).all()
            return posts, total

    def get_post(self, post_id: int) -> Optional[Post]:
        with self.SessionLocal() as session:
            return (
                session.query(Post)
                .options(selectinload(Post.post_tags).selectinload(PostTag.tag))
                .filter(Post.id == post_id)
                .one_or_none()
            )

    def random_posts(
        self,
        *,
        limit: int = 1,
        include_tags: Optional[Sequence[str]] = None,
        ratings: Optional[Sequence[str]] = None,
        min_score: Optional[int] = None,
    ) -> List[Post]:
        posts, _ = self.list_posts(
            include_tags=include_tags,
            ratings=ratings,
            min_score=min_score,
            limit=limit,
            random_order=True,
        )
        return posts


    def get_random_posts(self, limit: int = 10, min_score: Optional[int] = None) -> list[Post]:
        """Get random posts with optional score filtering."""
        if self.max_id == -1:
            with self.SessionLocal() as session:
                self.max_id = session.query(func.max(Post.id)).scalar() or 0

        if self.max_id == 0:
            raise Exception("No posts available in the database.")

        attempts = 0 # cap number of tries
        random_ids: Set[int] = set()
        while len(random_ids) < limit and attempts < 2 * limit:
            random_ids.add(random.randint(0, self.max_id))
            attempts = attempts+1

        with self.SessionLocal() as session:
            query = session.query(Post).filter(Post.seq_id.in_(random_ids))

            if min_score is not None:
                query = query.filter(Post.score > min_score)

            return query.order_by(func.random()).limit(limit).all()


    def get_tag(self, name: str) -> Optional[Tag]:
        with self.SessionLocal() as session:
            return session.query(Tag).filter(Tag.name == name).one_or_none()

    def search_tags(
        self,
        *,
        search: Optional[str] = None,
        category: Optional[int] = None,
        min_count: Optional[int] = None,
        limit: int = 50,
        offset: int = 0,
        sort_by: str = "count",
        descending: bool = True,
    ) -> Tuple[List[Tag], int]:
        limit = max(1, min(limit, 200))
        offset = max(0, offset)

        with self.SessionLocal() as session:
            query = session.query(Tag)

            if search:
                query = query.filter(Tag.name.ilike(f"%{search}%"))
            if category is not None:
                query = query.filter(Tag.category == category)
            if min_count is not None:
                query = query.filter(Tag.count >= min_count)

            total = int(query.with_entities(func.count()).scalar() or 0)

            sort_columns = {
                "count": Tag.count,
                "name": Tag.name,
                "avg_score": Tag.avg_score,
            }
            column = sort_columns.get(sort_by, Tag.count)
            direction = column.desc() if descending else column.asc()
            query = query.order_by(direction)

            tags = query.offset(offset).limit(limit).all()
            return tags, total


# TODO: Top artists
