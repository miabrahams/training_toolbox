from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Any, Dict, Iterable
from pathlib import Path
from collections import Counter

from src.lib.prompt_parser import clean_prompt
from src.lib.comfy_schemas.comfy_analysis_v2 import (
    extract_from_prompt,
    DEFAULT_SCHEMA_PATH,
)

from src.lib.database import TagDatabase
from .utils import noCallback
from .prompt_data import PromptData

MAX_FAILED = 1000 # Maximum allowed failed extractions before stopping processing

@dataclass
class ProcessingStats:
    processed: int = 0
    failed_extract: int = 0
    errors: int = 0


class PromptProcessor:
    """Controller that reads pending prompts and writes processed text back."""

    def __init__(self, db: TagDatabase):
        self.db = db

    def _iter_schema_paths(self) -> Iterable[Path]:
        """Yield available schema files, preferring newer versions first.

        Looks in the bundled schemas directory next to comfy_analysis_v2.
        """
        schemas_dir = DEFAULT_SCHEMA_PATH.parent
        if not schemas_dir.exists():
            return []

        # Prefer files like schema_v5.yml over schema_v3.yml by sorting reverse
        paths = sorted(schemas_dir.glob("schema_*.yml"), key=lambda p: p.name, reverse=True)
        return paths

    def _try_extract_fields(self, prompt_graph: Dict[str, Any]) -> Dict[str, Any] | None:
        """Attempt to extract fields using the first matching schema.

        Returns the extracted dict on success, or None when no schema matches.
        Raises on non-validation errors (e.g., unexpected runtime errors).
        """
        last_schema_error: Exception | None = None
        for schema_path in self._iter_schema_paths():
            try:
                return extract_from_prompt(prompt_graph, schema_path)
            except ValueError as ve:
                # Schema mismatch/validation error – try next schema
                last_schema_error = ve
                continue
        # No schema matched
        return None

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
                extracted = self._try_extract_fields(prompt)
                if not extracted:
                    # No schema matched – don't write anything
                    stats.failed_extract += 1
                    continue

                positive = extracted.get("positive_prompt")
                if not positive:
                    # Treat missing positive as schema failure (shouldn't happen if schema matched)
                    stats.failed_extract += 1
                    continue

                cleaned = clean_prompt(positive)

                # Map extracted outputs to DB columns where available
                extras: Dict[str, Any] = {}

                def to_int(v: Any) -> int | None:
                    try:
                        return int(v) if v is not None and str(v) != "" else None
                    except Exception:
                        return None

                def to_float(v: Any) -> float | None:
                    try:
                        return float(v) if v is not None and str(v) != "" else None
                    except Exception:
                        return None

                def to_bool(v: Any) -> bool | None:
                    if v is None:
                        return None
                    if isinstance(v, bool):
                        return v
                    s = str(v).strip().lower()
                    if s in {"1", "true", "yes", "on"}:
                        return True
                    if s in {"0", "false", "no", "off"}:
                        return False
                    return None
                # Direct field mappings present in PromptFields
                # String-like fields
                for k in [
                    "negative_prompt",
                    "loras",
                    "sampler_name",
                    "scheduler",
                    "ip_image",
                    "checkpoint",
                    "aspect_ratio",
                ]:
                    val = extracted.get(k)
                    if val is not None:
                        extras[k] = val

                # Numeric fields
                steps = to_int(extracted.get("steps"))
                if steps is not None:
                    extras["steps"] = steps
                seed = to_int(extracted.get("seed"))
                if seed is not None:
                    extras["seed"] = seed
                cfg_val = to_float(extracted.get("cfg"))
                if cfg_val is not None:
                    extras["cfg"] = cfg_val
                ip_weight = to_float(extracted.get("ip_weight"))
                if ip_weight is not None:
                    extras["ip_weight"] = ip_weight

                # Booleans
                ip_enabled = to_bool(extracted.get("ip_enabled"))
                if ip_enabled is not None:
                    extras["ip_enabled"] = ip_enabled
                rescale_cfg = to_bool(extracted.get("rescale_cfg"))
                if rescale_cfg is not None:
                    extras["rescale_cfg"] = rescale_cfg
                perp_neg = to_bool(extracted.get("perp_neg"))
                if perp_neg is not None:
                    extras["perp_neg"] = perp_neg
                swap_dimensions = to_bool(extracted.get("swap_dimensions"))
                if swap_dimensions is not None:
                    extras["swap_dimensions"] = swap_dimensions

                self.db.upsert_prompt_text(file_path, positive, cleaned, **extras)
                stats.processed += 1
            except json.JSONDecodeError:
                stats.errors += 1
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                # Unexpected failure – count as error
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