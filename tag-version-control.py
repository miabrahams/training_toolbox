import os
import sqlite3
from datetime import datetime
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

class CaptionVersionControl:
    def __init__(self, db_path="caption_versions.db"):
        """Initialize the version control system with a SQLite database."""
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Create the necessary database tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS caption_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT,
                    content TEXT,
                    content_hash TEXT,
                    timestamp DATETIME,
                    version INTEGER,
                    comment TEXT
                )
            ''')
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_file_path
                ON caption_versions(file_path)
            ''')
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_timestamp
                ON caption_versions(timestamp)
            ''')

    def calculate_hash(self, content: str) -> str:
        """Calculate SHA-256 hash of content for efficient comparison."""
        return hashlib.sha256(content.encode()).hexdigest()

    def bulk_add_versions(self, file_paths: List[str], comment: str = "") -> Tuple[int, int]:
        """
        Add multiple caption files to version control.

        Returns:
            Tuple[int, int]: (number of files added, number of files skipped)
        """
        added = 0
        skipped = 0

        # First, read all files and calculate hashes
        file_contents: Dict[str, Tuple[str, str]] = {}

        for file_path in tqdm(file_paths, desc="Reading files"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                file_contents[file_path] = (content, self.calculate_hash(content))
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue

        # Batch insert into database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Get current versions for all files
            placeholders = ','.join('?' * len(file_contents))
            cursor.execute(f'''
                SELECT file_path, content_hash, version
                FROM caption_versions
                WHERE file_path IN ({placeholders})
                AND version = (
                    SELECT MAX(version)
                    FROM caption_versions AS v2
                    WHERE v2.file_path = caption_versions.file_path
                )
            ''', list(file_contents.keys()))

            current_versions = {row[0]: (row[1], row[2]) for row in cursor.fetchall()}

            # Prepare batch insert data
            timestamp = datetime.now().isoformat()
            values = []

            for file_path, (content, content_hash) in file_contents.items():
                current = current_versions.get(file_path, (None, 0))
                if current[0] != content_hash:  # New content
                    values.append((
                        file_path,
                        content,
                        content_hash,
                        timestamp,
                        current[1] + 1,
                        comment
                    ))
                    added += 1
                else:
                    skipped += 1

            if values:
                cursor.executemany('''
                    INSERT INTO caption_versions
                    (file_path, content, content_hash, timestamp, version, comment)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', values)

            conn.commit()

        return added, skipped

    def bulk_restore_versions(self, file_paths: List[str],
                            versions: Optional[Dict[str, int]] = None,
                            timestamp: Optional[str] = None) -> Dict[str, bool]:
        """
        Restore multiple files to specific versions or to their state at a given timestamp.

        Args:
            file_paths: List of file paths to restore
            versions: Dictionary mapping file paths to specific versions to restore
            timestamp: ISO format timestamp to restore files to their state at that time

        Returns:
            Dict[str, bool]: Mapping of file paths to restoration success status
        """
        results = {}

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            for file_path in tqdm(file_paths, desc="Restoring files"):
                try:
                    if versions and file_path in versions:
                        # Restore specific version
                        cursor.execute('''
                            SELECT content
                            FROM caption_versions
                            WHERE file_path = ? AND version = ?
                        ''', (file_path, versions[file_path]))
                    elif timestamp:
                        # Restore version at timestamp
                        cursor.execute('''
                            SELECT content
                            FROM caption_versions
                            WHERE file_path = ?
                            AND timestamp <= ?
                            ORDER BY timestamp DESC LIMIT 1
                        ''', (file_path, timestamp))
                    else:
                        # Restore latest version
                        cursor.execute('''
                            SELECT content
                            FROM caption_versions
                            WHERE file_path = ?
                            ORDER BY version DESC LIMIT 1
                        ''', (file_path,))

                    result = cursor.fetchone()
                    if result:
                        os.makedirs(os.path.dirname(file_path), exist_ok=True)
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(result[0])
                        results[file_path] = True
                    else:
                        results[file_path] = False
                except Exception as e:
                    print(f"Error restoring {file_path}: {e}")
                    results[file_path] = False

        return results

    def bulk_get_versions(self, file_paths: List[str]) -> Dict[str, List[Tuple]]:
        """Get version histories for multiple files."""
        results = {}

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            placeholders = ','.join('?' * len(file_paths))
            cursor.execute(f'''
                SELECT file_path, version, timestamp, comment, content
                FROM caption_versions
                WHERE file_path IN ({placeholders})
                ORDER BY file_path, version DESC
            ''', file_paths)

            for row in cursor.fetchall():
                file_path = row[0]
                if file_path not in results:
                    results[file_path] = []
                results[file_path].append(row[1:])

        return results

    def scan_and_add_captions(self, root_folder: str, comment: str = "") -> Tuple[int, int]:
        """Recursively scan a folder and add all caption files to version control."""
        file_paths = list(Path(root_folder).rglob("*.txt"))
        return self.bulk_add_versions([str(p) for p in file_paths], comment)

def main():
    """Command-line interface for caption version control."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Version control system for caption files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Add all caption files from a directory
  python script.py add /path/to/captions --comment "Initial import"

  # List versions of specific files
  python script.py list /path/to/file1.txt /path/to/file2.txt

  # Restore specific files to their latest versions
  python script.py restore /path/to/file1.txt /path/to/file2.txt

  # Restore files to a specific version
  python script.py restore /path/to/file1.txt --version 2

  # Restore files to their state at a specific time
  python script.py restore /path/to/file1.txt --timestamp "2024-03-15T12:00:00"
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Add command
    add_parser = subparsers.add_parser('add', help='Add files to version control')
    add_parser.add_argument('path', help='File or directory path')
    add_parser.add_argument('--comment', '-c', help='Version comment', default='')
    add_parser.add_argument('--no-recurse', '-nr', action='store_true',
                           help='Do not recurse into subdirectories')

    # List command
    list_parser = subparsers.add_parser('list', help='List file versions')
    list_parser.add_argument('files', nargs='+', help='File paths to list versions for')

    # Restore command
    restore_parser = subparsers.add_parser('restore', help='Restore file versions')
    restore_parser.add_argument('files', nargs='+', help='File paths to restore')
    restore_parser.add_argument('--version', '-v', type=int,
                              help='Specific version to restore')
    restore_parser.add_argument('--timestamp', '-t',
                              help='Timestamp to restore to (ISO format)')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    vc = CaptionVersionControl()

    try:
        if args.command == 'add':
            path = Path(args.path)
            if path.is_file():
                added, skipped = vc.bulk_add_versions([str(path)], args.comment)
                print(f"Added {added} files, skipped {skipped} unchanged files")
            elif path.is_dir():
                if args.no_recurse:
                    files = [str(f) for f in path.glob("*.txt")]
                    added, skipped = vc.bulk_add_versions(files, args.comment)
                else:
                    # Only add .txt files in the root directory
                    added, skipped = vc.scan_and_add_captions(str(path), args.comment)
                print(f"Added {added} files, skipped {skipped} unchanged files")
            else:
                print(f"Error: Path '{args.path}' does not exist")

        elif args.command == 'list':
            histories = vc.bulk_get_versions(args.files)
            for file_path, versions in histories.items():
                print(f"\nVersions for {file_path}:")
                if not versions:
                    print("  No versions found")
                    continue
                for version, timestamp, comment, _ in versions:
                    print(f"  Version {version:2d} - {timestamp} - {comment or 'No comment'}")

        elif args.command == 'restore':
            if args.version is not None:
                # Restore specific version for all files
                versions = {file: args.version for file in args.files}
                results = vc.bulk_restore_versions(args.files, versions=versions)
            else:
                # Restore to timestamp or latest version
                results = vc.bulk_restore_versions(args.files, timestamp=args.timestamp)

            for file_path, success in results.items():
                status = "Success" if success else "Failed"
                version_info = f"version {args.version}" if args.version else \
                             f"timestamp {args.timestamp}" if args.timestamp else \
                             "latest version"
                print(f"Restored {file_path} to {version_info}: {status}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()