#!/usr/bin/env python3
"""CLI for prompt database maintenance.

Usage examples:
    # Process pending prompts into prompt_fields
    python cli.py --db data/prompts.sqlite

    # Reset (drop and recreate) prompt_fields table
    python cli.py reset_prompt_fields --db data/prompts.sqlite
"""

import argparse
from pathlib import Path

from lib.database import TagDatabase, PromptFields
from sqlalchemy import create_engine, select, insert
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.sql.schema import Table as SATable
from typing import cast
from src.tag_analyzer.processor import PromptProcessor
from lib.config import get_settings


def reset_prompt_fields(db: TagDatabase):
    """Drop and recreate the prompt_fields table."""
    db.drop_prompt_fields()
    db.ensure_schema()
    print("Prompt fields table reset.")


def export_prompt_fields(db: TagDatabase, out_path: Path):
    """Export the prompt_fields table (schema + data) to a separate SQLite DB.

    - Creates a new database at out_path if it doesn't exist
    - Creates the prompt_fields table (with indexes)
    - Copies all rows from the source database
    """
    # Initialize destination engine and session
    dest_engine = create_engine(f"sqlite:///{out_path}", future=True)
    DestSession = sessionmaker(bind=dest_engine, future=True, class_=Session)

    # Create only the prompt_fields table
    pf_table = cast(SATable, PromptFields.__table__)
    pf_table.create(bind=dest_engine, checkfirst=True)

    # Copy all rows from source to destination
    with db.SessionLocal() as src_sess, DestSession.begin() as dest_sess:
        rows = src_sess.execute(select(PromptFields)).scalars().all()
        if not rows:
            print("No rows found in prompt_fields to export.")
            return

        # Prepare bulk payload preserving column values, including id
        cols = [c.name for c in PromptFields.__table__.columns]
        payload = []
        for r in rows:
            payload.append({c: getattr(r, c) for c in cols})

        dest_sess.execute(insert(PromptFields), payload)
        print(f"Exported {len(payload)} rows to {out_path}")


def main():
    settings = get_settings()
    default_db_path = settings.get("ui.defaults.db_path") or settings.get("tools.check_db.default_path")
    default_db = Path(default_db_path).expanduser() if default_db_path else Path("data/prompts.sqlite")

    parser = argparse.ArgumentParser(description="Prompt DB maintenance and processing")
    subparsers = parser.add_subparsers(dest="command")

    # Shared --db arg at the root level for simplicity
    parser.add_argument(
        "--db", dest="db_path", type=Path, default=default_db, help="Path to SQLite database"
    )

    _ = subparsers.add_parser('reset_prompt_fields')
    export_parser = subparsers.add_parser('export_prompt_fields')
    export_parser.add_argument('--out', dest='out_path', type=Path, required=True, help='Output SQLite file path')

    args = parser.parse_args()

    db = TagDatabase(args.db_path)

    progress = lambda x, y: print(f"Progress: {x*100:.1f}% - {y}")


    # Dispatch based on command; default to processing when no command given
    match args.command:
        case "reset_prompt_fields":
            reset_prompt_fields(db)
            return
        case "export_prompt_fields":
            export_prompt_fields(db, args.out_path)
            return
        case _:
            # Default: ensure schema and process pending prompts
            processor = PromptProcessor(db)
            stats = processor.process_new_prompts(progress)
            print(f"Processed: {stats.processed}, failed_extract: {stats.failed_extract}, errors: {stats.errors}")


if __name__ == "__main__":
    main()
