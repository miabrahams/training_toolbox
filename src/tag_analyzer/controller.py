from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Optional, Tuple
from pathlib import Path

from lib.prompt_parser import clean_prompt
from lib.comfy_analysis import extract_positive_prompt

from .database import TagDatabase
from .utils import noCallback


@dataclass
class ProcessingStats:
    processed: int = 0
    failed_extract: int = 0
    errors: int = 0


class PromptProcessor:
    """Controller that reads pending prompts and writes processed text back."""

    def __init__(self, db: TagDatabase):
        self.db = db

    def process_pending(self, progress: Callable = noCallback) -> ProcessingStats:
        stats = ProcessingStats()

        pending = self.db.get_pending_prompts()
        total = len(pending)
        if total == 0:
            progress(1.0, "No pending prompts to process")
            return stats

        for i, (file_path, prompt_json) in enumerate(pending, start=1):
            # Update progress occasionally
            if total > 0:
                progress(min(0.99, i / total), f"Processing prompts {i}/{total}")

            try:
                prompt = json.loads(prompt_json)
                try:
                    positive = extract_positive_prompt(prompt)
                except Exception:
                    # Could not extract a positive prompt; mark unprocessed for audit
                    stats.failed_extract += 1
                    self.db.mark_unprocessed(file_path)
                    continue

                if not positive:
                    stats.failed_extract += 1
                    self.db.mark_unprocessed(file_path)
                    continue

                cleaned = clean_prompt(positive)
                self.db.upsert_prompt_text(file_path, positive, cleaned)
                stats.processed += 1
            except Exception:
                stats.errors += 1
                self.db.mark_unprocessed(file_path)

        progress(1.0, f"Processed {stats.processed}, failed {stats.failed_extract}, errors {stats.errors}")
        return stats

    def initialize_prompt_data(self, db_path: Path, progress: Callable = noCallback):
        """End-to-end flow used by UI/CLI without embedding SQL here.

        - Ensures schema
        - Optional diagnostics
        - Processes pending prompts
        - Loads processed prompts using TagDatabase
        Returns (PromptData, TagDatabase)
        """
        from .prompt_data import PromptData  # lazy import to avoid cycles

        progress(0.05, "Opening database...")
        db = self.db if self.db else TagDatabase(db_path)

        if not db.has_table("prompt_texts"):
            print("Creating 'prompt_texts' table...")
            db.ensure_schema()

        if db.has_table("prompts"):
            count = db.count_rows("prompts")
            print(f"Found {count} entries in the prompts table")
        else:
            print("WARNING: 'prompts' table does not exist in the database!")

        progress(0.2, "Processing pending prompts...")
        self.process_pending(progress)

        progress(0.7, "Loading processed prompts...")
        prompts_counter, image_paths = db.load_prompts()
        prompt_texts = list(prompts_counter.keys())
        progress(0.95, f"Loaded {len(prompt_texts)} unique prompts")

        return PromptData(prompt_texts, prompts_counter, image_paths), db
