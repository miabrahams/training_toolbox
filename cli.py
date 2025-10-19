#!/usr/bin/env python3
"""CLI to process pending ComfyUI prompts into prompt_texts table.

Usage:
  python tools/update_prompt_texts.py --db data/prompts.sqlite
"""

from pathlib import Path
import argparse

from src.tag_analyzer.database import TagDatabase
from src.tag_analyzer.controller import PromptProcessor


def main():
    parser = argparse.ArgumentParser(description="Process pending prompts into prompt_texts")
    parser.add_argument("--db", dest="db_path", type=Path, default=Path("data/prompts.sqlite"), help="Path to SQLite database")
    args = parser.parse_args()

    db = TagDatabase(args.db_path)
    processor = PromptProcessor(db)

    def progress(x, y):
        return print(f"Progress: {x*100:.1f}% - {y}")

    stats = processor.process_pending(progress)
    print(f"Processed: {stats.processed}, failed_extract: {stats.failed_extract}, errors: {stats.errors}")


if __name__ == "__main__":
    main()