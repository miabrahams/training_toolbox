from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    String,
    Text,
    create_engine,
    func,
    inspect,
    select,
    text as sql_text,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    sessionmaker,
    Session,
)
from collections import Counter


class Base(DeclarativeBase):
    pass


class Prompt(Base):
    __tablename__ = "prompts"

    file_path: Mapped[str] = mapped_column(String, primary_key=True)
    prompt: Mapped[str] = mapped_column(Text, nullable=False)

    prompt_text: Mapped["PromptText"] = relationship(
        back_populates="prompt", uselist=False, cascade="all, delete-orphan"
    )


class PromptText(Base):
    __tablename__ = "prompt_texts"
    file_path: Mapped[str] = mapped_column(String, ForeignKey("prompts.file_path"), primary_key=True)
    positive_prompt: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    cleaned_prompt: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    processed: Mapped[Optional[bool]] = mapped_column(Boolean, default=False, nullable=True)
    last_updated: Mapped[datetime] = mapped_column(DateTime, server_default=func.current_timestamp())

    prompt: Mapped["Prompt"] = relationship(back_populates="prompt_text")


Index("idx_cleaned_prompt", PromptText.cleaned_prompt)


class TagDatabase:
    """SQLAlchemy-based repository for prompt analysis data."""

    def __init__(self, db_path: Path = Path("data/prompts.sqlite")):
        self.db_path = db_path
        self.engine = create_engine(f"sqlite:///{db_path}", future=True)
        self.SessionLocal = sessionmaker(bind=self.engine, future=True, class_=Session)

    def ensure_schema(self):
        Base.metadata.create_all(self.engine)

    def has_table(self, table_name: str) -> bool:
        return inspect(self.engine).has_table(table_name)

    def count_rows(self, table_name: str) -> int:
        with self.engine.begin() as conn:
            return int(conn.execute(sql_text(f"SELECT COUNT(*) FROM {table_name}")).scalar() or 0)

    def drop_prompt_texts(self) -> None:
        """Drop the prompt_texts table (fast rebuild path when changing schema)."""
        with self.engine.begin() as conn:
            conn.execute(sql_text("DROP TABLE IF EXISTS prompt_texts"))

    def get_pending_prompts(self) -> List[Tuple[str, str]]:
        """Return list of (file_path, prompt_json) needing processing."""
        with self.SessionLocal() as session:
            p = Prompt
            pt = PromptText
            stmt = (
                select(p.file_path, p.prompt)
                .select_from(p)
                .outerjoin(pt, p.file_path == pt.file_path)
                .where((pt.file_path.is_(None)) | (pt.processed.is_(None)) | (pt.processed == False))
            )
            return [(fp, pr) for fp, pr in session.execute(stmt).all()]

    def upsert_prompt_text(self, file_path: str, positive_prompt: str, cleaned_prompt: str):
        with self.SessionLocal.begin() as session:
            existing: PromptText | None = session.get(PromptText, file_path)
            if existing:
                existing.positive_prompt = positive_prompt
                existing.cleaned_prompt = cleaned_prompt
                existing.processed = True
                existing.last_updated = func.current_timestamp()
            else:
                session.add(
                    PromptText(
                        file_path=file_path,
                        positive_prompt=positive_prompt,
                        cleaned_prompt=cleaned_prompt,
                        processed=True,
                    )
                )

    def mark_unprocessed(self, file_path: str):
        with self.SessionLocal.begin() as session:
            existing: PromptText | None = session.get(PromptText, file_path)
            if existing:
                existing.processed = False
                existing.last_updated = func.current_timestamp()
            else:
                session.add(PromptText(file_path=file_path, processed=False))

    def load_prompts(self) -> Tuple[Counter, Dict[str, str]]:
        """
        Load processed cleaned prompts and a prompt->image mapping.
        Returns tuple: (Counter-like dict of cleaned prompt counts, mapping cleaned->file_path)
        """
        with self.SessionLocal() as session:
            pt = PromptText
            rows = session.execute(select(pt.cleaned_prompt, pt.file_path).where(pt.processed == True)).all()
            cleaned_prompts = [r[0] for r in rows]
            counts: Counter = Counter(cleaned_prompts)
            image_paths = {r[0]: r[1] for r in rows}
            return counts, image_paths

    def _escape_like(self, value: str) -> str:
        """Escape %, _ and backslash for a SQL LIKE pattern (using \\ as escape)."""
        return (
            value.replace("\\", "\\\\")
            .replace("%", r"\%")
            .replace("_", r"\_")
        )

    def search_positive_prompts(self, query: str, limit: int = 50):
        if not query:
            return []
        with self.SessionLocal() as session:
            pt = PromptText
            escaped = self._escape_like(query)
            like = f"%{escaped}%"
            grouped = (
                select(pt.positive_prompt, func.max(pt.last_updated).label("max_updated"))
                .where(pt.positive_prompt.like(like, escape="\\"))
                .group_by(pt.positive_prompt)
                .subquery()
            )
            stmt = (
                select(pt.positive_prompt, pt.file_path)
                .join(
                    grouped,
                    (pt.positive_prompt == grouped.c.positive_prompt)
                    & (pt.last_updated == grouped.c.max_updated),
                )
                .where(pt.positive_prompt.like(like, escape="\\"))
                .order_by(pt.last_updated.desc())
                .limit(limit)
            )
            return [
                {"file_path": row[1], "positive_prompt": row[0]} for row in session.execute(stmt).all()
            ]

    def get_positive_prompts(self, file_paths: List[str]) -> Dict[str, str]:
        if not file_paths:
            return {}
        with self.SessionLocal() as session:
            pt = PromptText
            rows = session.execute(select(pt.file_path, pt.positive_prompt).where(pt.file_path.in_(file_paths))).all()
            return {fp: pos for fp, pos in rows}

    def get_image_paths(self, prompt_texts: List[str]) -> Dict[str, str]:
        if not prompt_texts:
            return {}
        with self.SessionLocal() as session:
            pt = PromptText
            rows = session.execute(select(pt.cleaned_prompt, pt.file_path).where(pt.cleaned_prompt.in_(prompt_texts))).all()
            return {cp: fp for cp, fp in rows}
