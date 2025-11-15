from typing import Dict, Any, Iterable
from pathlib import Path

from src.lib.errors import ExtractionFailedError
from .comfy_analysis_v2 import (
    extract_from_prompt,
    DEFAULT_SCHEMA_PATH,
)

from ...schemas.extracted_prompt import ExtractedPrompt

def extract_from_json(prompt: Dict[str, Any]) -> ExtractedPrompt:
    try:
        extracted = try_extract_fields(prompt)
    except ValueError as e:
        raise ExtractionFailedError from e
    positive = extracted.get("positive_prompt")
    if not positive:
        # Treat missing positive as schema failure (any crazy edge cases without a positive are fine to skip)
        raise ExtractionFailedError("no positive prompt found")
    return ExtractedPrompt(**extracted)

def try_extract_fields(prompt_graph: Dict[str, Any]) -> Dict[str, Any]:
    """Attempt to extract fields using the first matching schema."""
    last_schema_error: Exception | None = None
    for schema_path in iter_schema_paths():
        try:
            return extract_from_prompt(prompt_graph, schema_path)
        except ValueError as ve:
            last_schema_error = ve
            continue
    if last_schema_error is not None:
        raise last_schema_error
    raise ValueError("should not reach here")


def iter_schema_paths() -> Iterable[Path]:
    """Iterate over available schema file paths in reverse numerical order (newest first)"""
    schemas_dir = DEFAULT_SCHEMA_PATH.parent
    if not schemas_dir.exists():
        raise FileNotFoundError(f"Schemas directory not found: {schemas_dir}")
    return sorted(schemas_dir.glob("schema_*.yml"), key=lambda p: p.name, reverse=True)

