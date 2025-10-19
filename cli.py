#!/usr/bin/env python3
"""CLI for prompt database maintenance.

Usage examples:
    # Process pending prompts into prompt_texts
    python cli.py --db data/prompts.sqlite

    # Reset (drop and recreate) prompt_texts table
    python cli.py reset_prompt_texts --db data/prompts.sqlite
"""

from pathlib import Path
import argparse

from src.tag_analyzer.database import TagDatabase
from src.tag_analyzer.processor import PromptProcessor


def main():
    parser = argparse.ArgumentParser(description="Prompt DB maintenance and processing")
    subparsers = parser.add_subparsers(dest="command")

    # Shared --db arg at the root level for simplicity
    parser.add_argument(
        "--db", dest="db_path", type=Path, default=Path("data/prompts.sqlite"), help="Path to SQLite database"
    )

    _ = subparsers.add_parser('reset_prompt_texts')

    args = parser.parse_args()

    db = TagDatabase(args.db_path)

    # Dispatch based on command; default to processing when no command given
    if args.command == "reset_prompt_texts":
        print(f"Dropping prompt_texts in {args.db_path}...")
        db.drop_prompt_texts()
        print("Done.")
        return

    # Default: process pending prompts
    processor = PromptProcessor(db)

    def progress(x, y):
        return print(f"Progress: {x*100:.1f}% - {y}")

    stats = processor.process_pending(progress)
    print(f"Processed: {stats.processed}, failed_extract: {stats.failed_extract}, errors: {stats.errors}")


if __name__ == "__main__":
    main()