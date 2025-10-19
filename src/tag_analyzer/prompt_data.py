from collections import Counter
from typing import Dict, Tuple, List
import os


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

    def get_image_path(self, prompt: str) -> str | None:
        """Get image path for a given prompt."""
        return self.image_paths.get(prompt)

    def get_file_name(self, prompt: str) -> str | None:
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
