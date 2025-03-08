from pathlib import Path
from collections import Counter
from typing import Dict, Tuple, Callable, Any

from .database import TagDatabase
from .utils import noCallback

class PromptData:
    """Container for prompt data used by the tag analyzer"""

    def __init__(self, prompt_texts=None, image_paths=None):
        """Initialize with prompt texts and image paths"""
        self.prompt_texts = list(prompt_texts) if prompt_texts else []
        self.image_paths = image_paths or {}

    @property
    def prompts(self):
        """Return list of prompt texts for compatibility with existing code"""
        return self.prompt_texts

def initialize_prompt_data(db_path: Path, progress: Callable = noCallback) -> Tuple[PromptData, TagDatabase]:
    """
    Initialize prompt data from database

    Args:
        db_path: Path to the database file
        progress: Progress callback function

    Returns:
        Tuple of (PromptData, TagDatabase)
    """
    progress(0.1, "Initializing database connection...")
    db = TagDatabase(db_path)

    # Ensure the prompt_texts table is up to date
    progress(0.3, "Updating prompt database...")
    db.update_prompt_texts()

    progress(0.5, "Loading prompts from database...")
    # This will load data from the prompt_texts table
    prompt_counter, image_paths = db.load_prompts()

    # Convert counter to list of unique prompts
    unique_prompts = list(prompt_counter.keys())

    progress(0.9, f"Loaded {len(unique_prompts)} unique prompts")

    # Create and return the PromptData object
    return PromptData(prompt_texts=unique_prompts, image_paths=image_paths), db