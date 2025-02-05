package database

import (
	"database/sql"
	"fmt"
)

type FilePrompt struct {
	Path     string `db:"file_path"`
	Prompt   string `db:"prompt"`
	Workflow string `db:"workflow"`
}

func WithDB(dbPath string, fn func(db *sql.DB) error) error {

	// Create (or open) the sqlite DB at the root.
	db, err := sql.Open("sqlite3", dbPath)
	if err != nil {
		return fmt.Errorf("failed to open sqlite db: %w", err)
	}
	defer db.Close()

	// Create table if not exists.
	createTable := `
    CREATE TABLE IF NOT EXISTS prompts (
        file_path TEXT PRIMARY KEY,
        prompt TEXT,
        workflow TEXT
    );`
	if _, err := db.Exec(createTable); err != nil {
		return fmt.Errorf("failed to create table: %w", err)
	}
	return fn(db)
}
