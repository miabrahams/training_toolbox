from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable
from pathlib import Path
from collections import Counter

from lib.prompt_parser import clean_prompt
from lib.comfy_schemas.comfy_analysis import extract_positive_prompt

from ...lib.database import TagDatabase
from .utils import noCallback
from .prompt_data import PromptData


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
            if total > 0:
                progress(min(0.99, i / total), f"Processing prompts {i}/{total}")

            try:
                prompt = json.loads(prompt_json)
                try:
                    positive = extract_positive_prompt(prompt)
                except Exception:
                    stats.failed_extract += 1
                    continue

                if not positive:
                    stats.failed_extract += 1
                    continue

                cleaned = clean_prompt(positive)
                self.db.upsert_prompt_text(file_path, positive, cleaned)
                stats.processed += 1
            except Exception:
                stats.errors += 1

        progress(1.0, f"Processed {stats.processed}, failed {stats.failed_extract}, errors {stats.errors}")
        return stats

    def process_new_prompts(self, progress: Callable = noCallback):
        """
        - Ensure schema
        - Process pending prompts
        """
        if not self.db.has_table("prompts"):
            raise Exception("'prompts' table does not exist in the database!")

        if not self.db.has_table("prompt_texts"):
            print("Creating 'prompt_texts' table...")
            self.db.ensure_schema()
        self.process_pending(progress)

    def load_prompts(self) -> PromptData:
        prompts = self.db.load_prompts()
        prompt_texts = [p[0] for p in prompts]
        return PromptData(
            prompt_texts=prompt_texts,
            image_paths={p[0]: p[1] for p in prompts},
            prompts_counter=Counter(prompt_texts)
        )