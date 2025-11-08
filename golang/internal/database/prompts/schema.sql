CREATE TABLE IF NOT EXISTS prompts (
    file_path TEXT PRIMARY KEY,
    prompt TEXT NOT NULL,
    workflow TEXT NOT NULL
);

CREATE TABLE prompt_texts (
    file_path TEXT UNIQUE NOT NULL PRIMARY KEY REFERENCES prompts(file_path),
    positive_prompt TEXT,
    cleaned_prompt TEXT,
    processed BOOLEAN DEFAULT 0,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)