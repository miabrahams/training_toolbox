import os
import json
import sqlite3
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter
from pathlib import Path

from .utils import extract_positive_prompt, clean_prompt

class TagDatabase:
    """Handles database operations for prompt analysis."""

    def __init__(self, db_path: Path = Path("data/prompts.sqlite")):
        """Initialize with database path."""
        self.db_path = db_path

    def _init_database(self):
        """Initialize database schema if needed."""
        if not self.db_path.exists():
            print(f"Database not found at {self.db_path}")
            return

        conn = sqlite3.connect(self.db_path)
        try:
            # Create prompt_texts table if it doesn't exist
            conn.execute('''
                CREATE TABLE IF NOT EXISTS prompt_texts (
                    id INTEGER PRIMARY KEY,
                    file_path TEXT UNIQUE,
                    original_prompt TEXT,
                    cleaned_prompt TEXT,
                    processed BOOLEAN DEFAULT 0,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_file_path ON prompt_texts(file_path)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_cleaned_prompt ON prompt_texts(cleaned_prompt)')
            conn.commit()
        finally:
            conn.close()



    def load_prompts(self) -> Tuple[Counter, Dict[str, str]]:
        """
        Load prompts from the database, updating prompt_texts table as needed.

        Returns:
            Tuple containing:
                - Counter of cleaned prompts
                - Dictionary mapping prompts to image paths
        """
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database file not found: {self.db_path}")

        conn = sqlite3.connect(self.db_path)
        try:
            # 1. First, check if we need to update the prompt_texts table
            self._update_prompt_texts_table(conn)

            # 2. Now load from the prompt_texts table
            cursor = conn.execute('SELECT cleaned_prompt, file_path FROM prompt_texts WHERE processed = 1')
            rows = cursor.fetchall()

            cleaned_prompts = [row[0] for row in rows]
            image_paths = {row[0]: row[1] for row in rows}

            print(f"Loaded {len(cleaned_prompts)} processed prompts from database")

            return Counter(cleaned_prompts), image_paths
        finally:
            conn.close()

    def _update_prompt_texts_table(self, conn):
        """Update prompt_texts table with any new or modified entries."""
        # Get existing file paths from prompt_texts table
        cursor = conn.execute('SELECT file_path FROM prompt_texts')
        existing_paths = {row[0] for row in cursor}

        # Get all file paths from the prompts table
        cursor = conn.execute('SELECT file_path FROM prompts')
        all_paths = {row[0] for row in cursor}

        # Identify new paths that need processing
        new_paths = all_paths - existing_paths

        if new_paths:
            print(f"Found {len(new_paths)} new files to process")
            processed_count = 0
            bad_images = []

            # Process new paths
            for file_path in new_paths:
                try:
                    # Get the prompt JSON for this file
                    cursor = conn.execute('SELECT prompt FROM prompts WHERE file_path = ?', (file_path,))
                    row = cursor.fetchone()
                    if not row:
                        continue

                    prompt_json = row[0]
                    prompt = json.loads(prompt_json)
                    positive = extract_positive_prompt(prompt)
                    cleaned = clean_prompt(positive)

                    # Insert into prompt_texts table
                    conn.execute(
                        'INSERT INTO prompt_texts (file_path, original_prompt, cleaned_prompt, processed) VALUES (?, ?, ?, 1)',
                        (file_path, positive, cleaned)
                    )
                    processed_count += 1
                except Exception as e:
                    bad_images.append(file_path)

            conn.commit()
            print(f"Processed {processed_count} new prompts, {len(bad_images)} failed")
            if bad_images:
                print(f"Bad images: {bad_images[:10]}{'...' if len(bad_images) > 10 else ''}")

    def get_image_paths(self, prompt_texts: List[str]) -> Dict[str, str]:
        """
        Get mapping of prompts to image file paths from the prompt_texts table.

        Args:
            prompt_texts: List of cleaned prompts to find image paths for

        Returns:
            Dictionary mapping prompts to image paths
        """
        if not prompt_texts:
            return {}

        conn = sqlite3.connect(self.db_path)
        image_paths = {}

        try:
            # Use placeholders for the IN clause
            placeholders = ','.join('?' for _ in prompt_texts)
            query = f'SELECT cleaned_prompt, file_path FROM prompt_texts WHERE cleaned_prompt IN ({placeholders})'

            cursor = conn.execute(query, prompt_texts)
            for row in cursor:
                image_paths[row[0]] = row[1]

            return image_paths
        finally:
            conn.close()
