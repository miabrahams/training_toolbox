import os
import json
import sqlite3
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter

from .utils import extract_positive_prompt, clean_prompt

class TagDatabase:
    """Handles database operations for prompt analysis."""

    def __init__(self, db_path: str = "data/prompts.sqlite"):
        """Initialize with database path."""
        self.db_path = db_path

    def load_prompts(self) -> Tuple[Counter, Dict[str, str]]:
        """
        Load prompts from the database.

        Returns:
            Tuple containing:
                - Counter of cleaned prompts
                - Dictionary mapping prompts to image paths
        """
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Database file not found: {self.db_path}")

        conn = sqlite3.connect(self.db_path)
        bad_images = []
        positives = []
        image_paths = {}

        print(f"Loading prompts from database: {self.db_path}")
        for row in conn.execute('SELECT file_path, prompt FROM prompts'):
            try:
                prompt = json.loads(row[1])
                filename = row[0]
                positive = extract_positive_prompt(prompt)
                cleaned = clean_prompt(positive)
                positives.append(cleaned)
                image_paths[cleaned] = filename
            except Exception:
                bad_images.append(row[0])
                pass

        print(f"{len(positives)} / {len(positives) + len(bad_images)}")
        if bad_images:
            print(f"Bad images: {bad_images[:10]}{'...' if len(bad_images) > 10 else ''}")

        conn.close()
        return Counter(positives), image_paths

    def get_image_paths(self, prompt_texts: List[str]) -> Dict[str, str]:
        """
        Get mapping of prompts to image file paths.

        Args:
            prompt_texts: List of prompts to find image paths for

        Returns:
            Dictionary mapping prompts to image paths
        """
        conn = sqlite3.connect(self.db_path)
        image_paths = {}

        for row in conn.execute('SELECT file_path, prompt FROM prompts'):
            try:
                prompt = json.loads(row[1])
                filename = row[0]
                positive = extract_positive_prompt(prompt)
                cleaned = clean_prompt(positive)
                if cleaned in prompt_texts:
                    image_paths[cleaned] = filename
            except Exception:
                pass

        conn.close()
        return image_paths
