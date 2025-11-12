from datetime import datetime
from typing import List, Optional, Set

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
    func
)

import random

from sqlalchemy.orm import sessionmaker, relationship, Session

from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

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

    tag_id: Mapped[BigInteger] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String, nullable=False, unique=True, index=True)
    category: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    count: Mapped[int] = mapped_column(BigInteger, nullable=False)
    avg_score: Mapped[float] = mapped_column(Double, nullable=False)

    post_tags: Mapped["PostTag"] = relationship(back_populates="tag")

    def __repr__(self):
        return f"<Tag(tag_id={self.tag_id}, name='{self.name}', category={self.category})>"


class Post(E6Base):
    """Posts table with metadata."""

    __tablename__ = "posts"

    id: Mapped[BigInteger] = mapped_column(BigInteger, primary_key=True)
    seq_id: Mapped[Integer] = mapped_column(BigInteger, nullable=False)
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
    post_tags: Mapped["PostTag"] = relationship(back_populates="post")

    def __repr__(self):
        return f"<Post(id={self.id}, rating='{self.rating}', score={self.score})>"


class PostTag(E6Base):
    """Junction table linking posts to tags."""

    __tablename__ = "post_tags"

    post_id: Mapped[BigInteger] = mapped_column(BigInteger, ForeignKey("posts.id"), primary_key=True)
    tag_id: Mapped[BigInteger] = mapped_column(BigInteger, ForeignKey("tags.tag_id"), primary_key=True)

    post: Mapped[List["Post"]] = relationship("Post", back_populates="post_tags")
    tag: Mapped[List["Tag"]] = relationship("Tag", back_populates="post_tags")

    def __repr__(self):
        return f"<PostTag(post_id={self.post_id}, tag_id={self.tag_id})>"


def connect_db(db_path) -> sessionmaker[Session]:
    engine = create_engine(f"duckdb:///{db_path}")
    E6Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)

def get_posts_with_tag(
    session: Session,
    tag_name: str,
    min_score: Optional[int] = None,
    limit: int = 10,
    random: bool = False,
):
    query = (
        session.query(Post)
        .join(PostTag, Post.id == PostTag.post_id)
        .join(Tag, PostTag.tag_id == Tag.tag_id)
        .filter(Tag.name == tag_name)
    )

    if min_score is not None:
        query = query.where(Post.score >= min_score)

    if random:
        query = query.order_by(func.random)

    if limit > 0:
        query = query.limit(limit)

    return query.all()


def get_posts_with_all_tags(
    session: Session,
    tag_names: list[str],
    exclude_tags: Optional[list[str]] = None,
    min_score: Optional[int] = None,
    min_favs: Optional[int] = None,
    limit: int = 10,
    random: bool = False,
) -> list[Post]:

    query = (
        session.query(Post)
        .join(PostTag, Post.id == PostTag.post_id)
        .join(Tag, PostTag.tag_id == Tag.tag_id)
        .filter(Tag.name.in_(tag_names))
    )

    # Filter out excluded tags if specified
    if exclude_tags:
        excluded_post_ids = (
            session.query(PostTag.post_id)
            .join(Tag, PostTag.tag_id == Tag.tag_id)
            .filter(Tag.name.in_(exclude_tags))
        )
        query = query.filter(~Post.id.in_(excluded_post_ids))

    # Apply score and fav filters
    if min_score is not None:
        query = query.filter(Post.score > min_score)

    if min_favs is not None:
        query = query.filter(Post.fav_count > min_favs)

    # Group by post and ensure all tags are present
    query = (
        query.group_by(Post.id)
        .having(func.count(func.distinct(Tag.name)) == len(tag_names))
    )

    if random:
        query = query.order_by(func.random())

    return query.limit(limit).all()


def count_posts_with_tag(session: Session, tag_name: str, min_score: Optional[int] = None) -> int:
    """Count posts matching tag criteria.

    Args:
        session: SQLAlchemy session
        tag_name: Name of the tag to filter by
        min_score: Minimum score threshold

    Returns:
        Count of matching posts
    """
    query = (
        session.query(Post)
        .join(PostTag, Post.id == PostTag.post_id)
        .join(Tag, PostTag.tag_id == Tag.tag_id)
        .filter(Tag.name == tag_name)
    )

    if min_score is not None:
        query = query.filter(Post.score > min_score)

    return query.count()

max_id = -1

def get_random_posts(session: Session, limit: int = 10, min_score: Optional[int] = None) -> list[Post]:
    """Get random posts with optional score filtering."""
    global max_id
    if max_id == -1:
        max_id = session.query(func.max(Post.id)).scalar() or 0

    attempts = 0 # cap number of tries
    random_ids: Set[int] = set()
    while len(random_ids) < limit and attempts < 2 * limit:
        random_ids.add(random.randint(0, max_id))
        attempts = attempts+1

    query = session.query(Post).filter(Post.seq_id.in_(random_ids))

    if min_score is not None:
        query = query.filter(Post.score > min_score)

    return query.order_by(func.random()).limit(limit).all()


# TODO: Top artists