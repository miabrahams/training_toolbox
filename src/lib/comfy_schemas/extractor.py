from typing import Dict, Any, Iterable
from pathlib import Path

from src.lib.errors import ExtractionFailedError
from src.schemas.prompt import ExtractedPrompt

from .comfy_analysis_v2 import (
    DEFAULT_SCHEMA_PATH,
    extract_from_prompt,
)


resolution_strings = {
  '1:1 square 1024x1024': (1024, 1024),
  '3:4 portrait 896x1152': (896, 1152),
  '5:8 portrait 832x1216': (832, 1216),
  '9:16 portrait 768x1344': (768, 1344),
  '9:16 portrait 768x1344': (768, 1344),
  '4:3 landscape 1152x896': (1152, 896),
  '3:2 landscape 1216x832': (1216, 832),
  '16:9 landscape 1344x768': (1344, 768),
}


def ar_to_dimensions(ar: str, swap: bool = False) -> tuple[int, int]:
    """Convert aspect ratio string to width and height values."""
    # pre-formatted resolution strings
    for res_str, (w, h) in resolution_strings.items():
        if ar == res_str:
            return (h, w) if swap else (w, h)

    # raw format "WxH"
    width_str, height_str = ar.lower().split("x")
    width = int(width_str)
    height = int(height_str)
    return (height, width) if swap else (width, height)


def extract_from_json(prompt: Dict[str, Any]) -> ExtractedPrompt:
    try:
        fields = try_extract_fields(prompt)
        if fields.aspect_ratio is not None:
            fields.width, fields.height = ar_to_dimensions(fields.aspect_ratio, swap=fields.swap_dimensions or False)
        return fields

    except ValueError as e:
        print("error: ", e)
        raise ExtractionFailedError from e

def try_extract_fields(prompt_graph: Dict[str, Any]) -> ExtractedPrompt:
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
