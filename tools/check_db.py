#!/usr/bin/env python
import sqlite3
import json
import sys
from pathlib import Path
import os

from src.lib.config import load_settings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.lib.comfy_schemas.comfy_analysis import extract_positive_prompt

def check_database(db_path):
    """Check database structure and sample data"""
    print(f"Checking database: {db_path}")

    conn: sqlite3.Connection | None = None

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # List all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        print(f"Tables in database: {[t[0] for t in tables]}")

        # Check prompts table
        try:
            cursor.execute("SELECT COUNT(*) FROM prompts")
            count = cursor.fetchone()[0]
            print(f"Prompts table contains {count} entries")

            # Get a sample
            cursor.execute("SELECT file_path, prompt FROM prompts LIMIT 3")
            samples = cursor.fetchall()
            for i, (path, prompt_json) in enumerate(samples):
                print(f"\nSample {i+1} - {path}:")
                try:
                    prompt_data = json.loads(prompt_json)
                    print(f"JSON Keys: {list(prompt_data.keys())}")

                    # Try extracting the positive prompt
                    positive = extract_positive_prompt(prompt_data)
                    print(f"Extracted positive prompt: {'<empty>' if not positive else positive[:100] + '...' if len(positive) > 100 else positive}")

                    # If positive is empty, look deeper
                    if not positive:
                        print("Detailed examination:")
                        for key, value in prompt_data.items():
                            if isinstance(value, dict):
                                print(f"  {key} (dict): {list(value.keys())}")
                            elif isinstance(value, list):
                                print(f"  {key} (list[{len(value)}]): {value[:2] if value else 'empty'}")
                            else:
                                print(f"  {key} ({type(value).__name__}): {value if len(str(value)) < 100 else str(value)[:100] + '...'}")

                except json.JSONDecodeError as e:
                    print(f"Not valid JSON: {e}")
                    print(f"Raw content: {prompt_json[:100]}")
                except Exception as e:
                    print(f"Error processing sample: {e}")
        except Exception as e:
            print(f"Error checking prompts table: {e}")

        # Check prompt_texts table
        try:
            cursor.execute("SELECT COUNT(*) FROM prompt_texts")
            count = cursor.fetchone()[0]
            print(f"\nPrompt_texts table contains {count} entries")

            # Get a sample
            cursor.execute("SELECT file_path, original_prompt, cleaned_prompt, processed FROM prompt_texts LIMIT 3")
            samples = cursor.fetchall()
            for i, (path, original, cleaned, processed) in enumerate(samples):
                print(f"\nSample {i+1} - {path}:")
                print(f"Processed: {bool(processed)}")
                print(f"Original prompt: {original[:100] + '...' if len(original) > 100 else original}")
                print(f"Cleaned prompt: {cleaned[:100] + '...' if len(cleaned) > 100 else cleaned}")
        except Exception as e:
            print(f"Error checking prompt_texts table: {e}")

    except Exception as e:
        print(f"Database error: {e}")
    finally:
        if conn:
            conn.close()

    print("Done")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        db_path = sys.argv[1]
    else:
        settings = load_settings()
        db_path = settings.get("tools.check_db.default_path") or "data/prompts.sqlite"

    check_database(Path(db_path))
