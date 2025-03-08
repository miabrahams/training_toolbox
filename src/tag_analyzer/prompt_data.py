from pathlib import Path
from typing import Tuple, Callable, List, Dict
from .database import TagDatabase
from .utils import noCallback
from collections import Counter


class PromptData:
    def __init__(self, prompt_texts: List[str], prompts: Counter, image_paths: Dict[str, str], data_dir: Path):
        self.prompt_texts = prompt_texts
        self.prompts = prompts
        self.image_paths = image_paths
        self.data_dir = data_dir


def initialize_prompt_data(db_path: Path, progress: Callable = noCallback) -> Tuple[PromptData, TagDatabase]:
    """Initialize prompt data from the database"""
    progress(0.1, "Initializing database...")

    # Initialize the database
    db = TagDatabase(db_path)

    # Load prompts and image paths from the database
    progress(0.5, "Loading prompts from database...")
    prompts_counter, image_paths = db.load_prompts()
    prompt_texts = list(prompts_counter.keys())

    # Create PromptData instance
    prompt_data = PromptData(
        prompt_texts=prompt_texts,
        prompts=prompts_counter,
        image_paths=image_paths,
        data_dir=db_path.parent
    )

    progress(1.0, "Prompt data loaded successfully")
    return prompt_data, db