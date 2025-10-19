import sqlite3
from typing import Dict, List, Tuple
from collections import Counter
from pathlib import Path


class TagDatabase:
    """SQLite repository for prompt analysis data (no business logic)."""

    def __init__(self, db_path: Path = Path("data/prompts.sqlite")):
        self.db_path = db_path

        if not self.db_path.exists():
            # Defer creation to caller; some commands may just inspect paths
            print(f"Database not found at {self.db_path}")
            return

        self.ensure_schema()

    def ensure_schema(self):
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS prompt_texts (
                    file_path TEXT PRIMARY KEY REFERENCES prompts(file_path),
                    positive_prompt TEXT,
                    cleaned_prompt TEXT,
                    processed BOOLEAN DEFAULT 0,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_cleaned_prompt ON prompt_texts(cleaned_prompt)"
            )
            conn.commit()
        finally:
            conn.close()

    def has_table(self, table_name: str) -> bool:
        if not self.db_path.exists():
            return False
        conn = sqlite3.connect(self.db_path)
        try:
            cur = conn.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,),
            )
            return cur.fetchone()[0] > 0
        finally:
            conn.close()

    def count_rows(self, table_name: str) -> int:
        if not self.db_path.exists():
            return 0
        conn = sqlite3.connect(self.db_path)
        try:
            cur = conn.execute(f"SELECT COUNT(*) FROM {table_name}")
            return int(cur.fetchone()[0])
        finally:
            conn.close()

    def get_pending_prompts(self) -> List[Tuple[str, str]]:
        """Return list of (file_path, prompt_json) needing processing."""
        if not self.db_path.exists():
            return []
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                """
                SELECT p.file_path, p.prompt
                FROM prompts p
                LEFT JOIN prompt_texts pt ON p.file_path = pt.file_path
                WHERE pt.file_path IS NULL OR pt.processed IS NULL OR pt.processed = 0
                """
            )
            return cursor.fetchall()
        finally:
            conn.close()

    def upsert_prompt_text(self, file_path: str, positive_prompt: str, cleaned_prompt: str):
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute("SELECT 1 FROM prompt_texts WHERE file_path = ?", (file_path,))
            exists = cursor.fetchone() is not None
            if exists:
                conn.execute(
                    """
                    UPDATE prompt_texts
                    SET positive_prompt = ?, cleaned_prompt = ?, processed = 1, last_updated = CURRENT_TIMESTAMP
                    WHERE file_path = ?
                    """,
                    (positive_prompt, cleaned_prompt, file_path),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO prompt_texts (file_path, positive_prompt, cleaned_prompt, processed)
                    VALUES (?, ?, ?, 1)
                    """,
                    (file_path, positive_prompt, cleaned_prompt),
                )
            conn.commit()
        finally:
            conn.close()

    def load_prompts(self) -> Tuple[Counter, Dict[str, str]]:
        """Load processed cleaned prompts and a prompt->image mapping."""
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database file not found: {self.db_path}")

        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                "SELECT cleaned_prompt, file_path FROM prompt_texts WHERE processed = 1"
            )
            rows = cursor.fetchall()
            cleaned_prompts = [row[0] for row in rows]
            image_paths = {row[0]: row[1] for row in rows}
            return Counter(cleaned_prompts), image_paths
        finally:
            conn.close()

    def search_positive_prompts(self, query: str, limit: int = 50):
        if not query:
            return []
        conn = sqlite3.connect(self.db_path)
        try:
            sql_query = (
                """
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
                """
            )
            search_term = f"%{query}%"
            cursor = conn.execute(sql_query, (search_term, search_term, limit))
            return [
                {"file_path": row[1], "positive_prompt": row[0]} for row in cursor.fetchall()
            ]
        finally:
            conn.close()

    def get_positive_prompts(self, file_paths: List[str]) -> Dict[str, str]:
        if not file_paths:
            return {}
        conn = sqlite3.connect(self.db_path)
        try:
            placeholders = ",".join("?" for _ in file_paths)
            query = f"SELECT file_path, positive_prompt FROM prompt_texts WHERE file_path IN ({placeholders})"
            cursor = conn.execute(query, file_paths)
            return {row[0]: row[1] for row in cursor}
        finally:
            conn.close()

    def get_image_paths(self, prompt_texts: List[str]) -> Dict[str, str]:
        if not prompt_texts:
            return {}
        conn = sqlite3.connect(self.db_path)
        try:
            placeholders = ",".join("?" for _ in prompt_texts)
            query = f"SELECT cleaned_prompt, file_path FROM prompt_texts WHERE cleaned_prompt IN ({placeholders})"
            cursor = conn.execute(query, prompt_texts)
            return {row[0]: row[1] for row in cursor}
        finally:
            conn.close()