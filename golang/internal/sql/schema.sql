CREATE TABLE IF NOT EXISTS prompts (
    file_path TEXT PRIMARY KEY,
    prompt TEXT NOT NULL,
    workflow TEXT NOT NULL
);

CREATE TABLE prompt_texts (
    id INTEGER PRIMARY KEY,
    file_path TEXT UNIQUE ,
    original_prompt TEXT,
    cleaned_prompt TEXT,
    processed BOOLEAN DEFAULT 0,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)