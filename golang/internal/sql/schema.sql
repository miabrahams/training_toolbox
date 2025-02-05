CREATE TABLE IF NOT EXISTS prompts (
    file_path TEXT PRIMARY KEY,
    prompt TEXT NOT NULL,
    workflow TEXT NOT NULL
);