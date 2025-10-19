#!/usr/bin/env python3
"""CLI for prompt database maintenance.

Usage examples:
    # Process pending prompts into prompt_fields
    python cli.py --db data/prompts.sqlite

    # Reset (drop and recreate) prompt_fields table
    python cli.py reset_prompt_fields --db data/prompts.sqlite
"""

from pathlib import Path
import argparse

from lib.database import TagDatabase
from src.tag_analyzer.processor import PromptProcessor


def reset_prompt_fields(db: TagDatabase):
    """Drop and recreate the prompt_fields table."""
    db.drop_prompt_fields()
    db.ensure_schema()
    print("Prompt fields table reset.")



def main():
    parser = argparse.ArgumentParser(description="Prompt DB maintenance and processing")
    subparsers = parser.add_subparsers(dest="command")

    # Shared --db arg at the root level for simplicity
    parser.add_argument(
        "--db", dest="db_path", type=Path, default=Path("data/prompts.sqlite"), help="Path to SQLite database"
    )

    _ = subparsers.add_parser('reset_prompt_fields')

    args = parser.parse_args()

    db = TagDatabase(args.db_path)

    progress = lambda x, y: print(f"Progress: {x*100:.1f}% - {y}")


    # Dispatch based on command; default to processing when no command given
    match args.command:
        case "reset_prompt_fields":
            reset_prompt_fields(db)
            return
        case _:
            # Default: ensure schema and process pending prompts
            processor = PromptProcessor(db)
            stats = processor.process_new_prompts(progress)
            print(f"Processed: {stats.processed}, failed_extract: {stats.failed_extract}, errors: {stats.errors}")


if __name__ == "__main__":
    main()