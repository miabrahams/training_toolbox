from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, List, Tuple, Optional
from collections import Counter


from src.db.prompt_database import PromptDatabase, PromptFields
from ..tags.utils import noCallback
from .prompt_data import PromptData
from src.lib.comfy_schemas.extractor import extract_from_json

MAX_FAILED = 1000 # Maximum allowed failed extractions before stopping processing

@dataclass
class ProcessingStats:
    processed: int = 0
    failed_extract: int = 0
    errors: int = 0


class PromptProcessor:
    """Controller that reads pending prompts and writes processed text back."""

    def __init__(self, db: PromptDatabase):
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
                extracted = extract_from_json(prompt)
                self.db.upsert_prompt_text(file_path, extracted)
                stats.processed += 1
            except json.JSONDecodeError:
                stats.errors += 1
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                stats.errors += 1

            if stats.failed_extract >= MAX_FAILED:
                print(f"Maximum failed extractions reached ({MAX_FAILED}). Stopping processing.")
                break

        progress(1.0, f"Processed {stats.processed}, failed {stats.failed_extract}, errors {stats.errors}")
        return stats

    def process_new_prompts(self, progress: Callable = noCallback) -> ProcessingStats:
        """
        - Ensure schema
        - Process pending prompts
        """
        if not self.db.has_table("prompts"):
            raise Exception("'prompts' table does not exist in the database!")

        # Always ensure schema (also applies lightweight migrations to add columns)
        if not self.db.has_table("prompt_fields"):
            print("Creating 'prompt_fields' table...")
        self.db.ensure_schema()

        return self.process_pending(progress)

    def load_prompts(self) -> PromptData:
        """Load processed prompts from the database into a PromptData object.

        Uses positive_prompt as the canonical text for now.
        """
        rows = self.db.load_prompts()  # List[Tuple[positive_prompt, file_path]]
        prompt_texts = [r[0] for r in rows]
        return PromptData(
            prompt_texts=prompt_texts,
            image_paths={r[0]: r[1] for r in rows},
            prompts_counter=Counter(prompt_texts),
        )

    def query_prompts(self, **filters) -> Tuple[List[PromptFields], int]:
        return self.db.query_prompt_fields(**filters)

    def get_prompt(self, prompt_id: int) -> Optional[PromptFields]:
        return self.db.get_prompt_by_id(prompt_id)

    def random_prompts(self, **kwargs) -> List[PromptFields]:
        return self.db.get_random_prompts(**kwargs)

    def stats(self) -> dict:
        return self.db.prompt_stats()
