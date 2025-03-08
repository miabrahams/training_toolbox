from pathlib import Path
from collections import Counter
from typing import Dict, Tuple, Callable, Any, List
import sqlite3
import os

from .database import TagDatabase
from .utils import noCallback

class PromptData:
    """Class to store and manage prompt data."""

    def __init__(self, prompt_texts: List[str], prompts_counter: Counter, image_paths: Dict[str, str]):
        """
        Initialize with prompt data.

        Args:
            prompt_texts: List of cleaned prompt texts
            prompts_counter: Counter of cleaned prompts
            image_paths: Dictionary mapping cleaned prompts to image paths
        """
        self.prompt_texts = prompt_texts
        self.prompts = prompts_counter
        self.image_paths = image_paths

    def get_image_path(self, prompt: str) -> str:
        """Get image path for a given prompt."""
        return self.image_paths.get(prompt)

    def get_file_name(self, prompt: str) -> str:
        """Get just the file name without the directory path."""
        path = self.get_image_path(prompt)
        if path:
            return os.path.basename(path)
        return None

    def find_prompts_by_path_fragment(self, path_fragment: str) -> List[str]:
        """
        Find prompts whose image path contains the given fragment.

        Args:
            path_fragment: A substring to look for in image paths

        Returns:
            List of prompts whose image path contains the fragment
        """
        matching_prompts = []
        for prompt, path in self.image_paths.items():
            if path_fragment in path:
                matching_prompts.append(prompt)
        return matching_prompts

    def get_most_common_prompts(self, n: int = 10) -> List[Tuple[str, int]]:
        """Get the n most common prompts with their counts."""
        return self.prompts.most_common(n)

    @classmethod
    def from_database(cls, db):
        """
        Create a PromptData instance from a TagDatabase.

        Args:
            db: A TagDatabase instance

        Returns:
            A new PromptData instance with data from the database
        """
        prompts_counter, image_paths = db.load_prompts()
        prompt_texts = list(prompts_counter.keys())
        return cls(prompt_texts, prompts_counter, image_paths)

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

    # Check if the database exists and has the expected tables
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check if the prompts table exists and has data
        cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='prompts'")
        if cursor.fetchone()[0] == 0:
            print("WARNING: 'prompts' table does not exist in the database!")
        else:
            cursor.execute("SELECT COUNT(*) FROM prompts")
            prompts_count = cursor.fetchone()[0]
            print(f"Found {prompts_count} entries in the prompts table")

        # Check if prompt_texts table exists
        cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='prompt_texts'")
        if cursor.fetchone()[0] == 0:
            print("WARNING: 'prompt_texts' table does not exist in the database!")
        else:
            cursor.execute("SELECT COUNT(*) FROM prompt_texts WHERE processed=1")
            texts_count = cursor.fetchone()[0]
            print(f"Found {texts_count} processed entries in the prompt_texts table")

        conn.close()
    except Exception as e:
        print(f"Error checking database tables: {e}")

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
    return PromptData(prompt_texts=unique_prompts, prompts_counter=prompt_counter, image_paths=image_paths), db