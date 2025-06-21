import json
import sqlite3
from typing import Dict, List, Tuple
from collections import Counter
from pathlib import Path

from lib.prompt_parser import clean_prompt
from lib.comfy_analysis import extract_positive_prompt

class TagDatabase:
    """Handles database operations for prompt analysis."""

    def __init__(self, db_path: Path = Path("data/prompts.sqlite")):
        """Initialize database schema if needed."""
        self.db_path = db_path

        if not self.db_path.exists():
            print(f"Database not found at {self.db_path}")
            return

        conn = sqlite3.connect(self.db_path)
        try:
            # Create prompt_texts table if it doesn't exist - now with positive_prompt column
            conn.execute('''
                CREATE TABLE IF NOT EXISTS prompt_texts (
                    file_path TEXT PRIMARY KEY REFERENCES prompts(file_path),
                    positive_prompt TEXT,
                    cleaned_prompt TEXT,
                    processed BOOLEAN DEFAULT 0,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_cleaned_prompt ON prompt_texts(cleaned_prompt)')
            conn.commit()

            # Ensure prompt_texts table is populated
            self._update_prompt_texts_table(conn)
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
        """Update prompt_texts table with any new or modified entries using SQL joins."""
        # Find new prompts that are not in prompt_texts
        cursor = conn.execute('''
            SELECT p.file_path, p.prompt
            FROM prompts p
            LEFT JOIN prompt_texts pt ON p.file_path = pt.file_path
            WHERE pt.file_path IS NULL
        ''')
        new_prompts = cursor.fetchall()

        # Find unprocessed prompts
        cursor = conn.execute('''
            SELECT p.file_path, p.prompt
            FROM prompts p
            LEFT JOIN prompt_texts pt ON p.file_path = pt.file_path
            WHERE pt.processed is NULL
        ''')
        new_prompts += cursor.fetchall()
        print(f"Found {len(new_prompts)} modified prompts to update")

        paths_to_process = new_prompts
        if not paths_to_process:
            return

        processed_count = 0
        bad_images = []
        failed_count = 0

        # Process new and modified paths
        for file_path, prompt_json in paths_to_process:
            try:
                # Parse the prompt JSON
                prompt = json.loads(prompt_json)

                # Extract positive prompt using the function from comfy_analysis
                try:
                    positive = extract_positive_prompt(prompt)
                except Exception as e:
                    print(f"Failed {file_path} - could not extract positive prompt", e)
                    failed_count += 1
                    continue

                # Check if we got a valid positive prompt
                if not positive or positive == "":
                    # Debug: print a bit more info about the failed prompt
                    print(f"Failed {file_path} - no positive prompt found")
                    if isinstance(prompt, dict):
                        print(f"  Available keys: {list(prompt.keys())}")
                    failed_count += 1
                    continue

                # Generate cleaned prompt from positive prompt
                cleaned = clean_prompt(positive)

                # Check if this is a new entry or an update
                cursor = conn.execute('SELECT 1 FROM prompt_texts WHERE file_path = ?', (file_path,))
                is_update = cursor.fetchone() is not None

                if is_update:
                    conn.execute(
                        '''UPDATE prompt_texts
                           SET positive_prompt = ?, cleaned_prompt = ?, processed = 1, last_updated = CURRENT_TIMESTAMP
                           WHERE file_path = ?''',
                        (positive, cleaned, file_path)
                    )
                else:
                    conn.execute(
                        '''INSERT INTO prompt_texts
                           (file_path, positive_prompt, cleaned_prompt, processed)
                           VALUES (?, ?, ?, 1)''',
                        (file_path, positive, cleaned)
                    )

                processed_count += 1
            except Exception as e:
                bad_images.append(file_path)
                print(f"Error processing {file_path}: {e}")

        conn.commit()
        print(f"Processed {processed_count} prompts, got nothing for {failed_count} prompts, {len(bad_images)} errored")
        if bad_images:
            print(f"Bad images: {bad_images[:10]}{'...' if len(bad_images) > 10 else ''}")

    def search_positive_prompts(self, query: str, limit: int = 50) -> List[Dict[str, str]]:
        """
        Search for unique prompts containing the query text in the positive_prompt field.
        Returns the most recent file_path for each unique prompt.
        """
        if not query:
            return []

        conn = sqlite3.connect(self.db_path)
        try:
            sql_query = '''
                SELECT pt.positive_prompt, pt.file_path
                FROM prompt_texts pt
                INNER JOIN (
                    SELECT positive_prompt, max(last_updated) as max_updated
                    FROM prompt_texts
                    WHERE positive_prompt LIKE ?
                    GROUP BY positive_prompt
                ) grouped
                ON pt.positive_prompt = grouped.positive_prompt AND pt.last_updated = grouped.max_updated
                WHERE pt.positive_prompt LIKE ?
                ORDER BY pt.last_updated DESC
                LIMIT ?
            '''
            search_term = f'%{query}%'
            cursor = conn.execute(sql_query, (search_term, search_term, limit))
            results = [{'file_path': row[1], 'positive_prompt': row[0]} for row in cursor.fetchall()]
            return results
        finally:
            conn.close()

    def update_prompt_texts(self):
        """
        Force an update of the prompt_texts table from the prompts table.
        Useful for manual calls to ensure database is up to date.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            self._update_prompt_texts_table(conn)
        finally:
            conn.close()

    def get_positive_prompts(self, file_paths: List[str]) -> Dict[str, str]:
        """
        Get the original positive prompts for given file paths.

        Args:
            file_paths: List of file paths to retrieve positive prompts for

        Returns:
            Dictionary mapping file paths to positive prompts
        """
        if not file_paths:
            return {}

        conn = sqlite3.connect(self.db_path)
        positive_prompts = {}

        try:
            placeholders = ','.join('?' for _ in file_paths)
            query = f'SELECT file_path, positive_prompt FROM prompt_texts WHERE file_path IN ({placeholders})'

            cursor = conn.execute(query, file_paths)
            for row in cursor:
                positive_prompts[row[0]] = row[1]

            return positive_prompts
        finally:
            conn.close()

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
