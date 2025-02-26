package database

import (
	"database/sql"
	"fmt"
	"os"
	"path/filepath"
)

// FilePrompt represents a prompt extracted from a file
type FilePrompt struct {
	Path     string
	Prompt   string
	Workflow string
}

// WithDB opens a database connection, initializes the schema, and calls the provided function
func WithDB(dbPath string, fn func(*sql.DB) error) error {
	// Ensure the directory exists
	if err := os.MkdirAll(filepath.Dir(dbPath), 0755); err != nil {
		return fmt.Errorf("failed to create database directory: %w", err)
	}

	db, err := sql.Open("sqlite3", dbPath)
	if err != nil {
		return fmt.Errorf("failed to open database: %w", err)
	}
	defer db.Close()

	// Initialize the schema
	if err := initSchema(db); err != nil {
		return fmt.Errorf("failed to initialize schema: %w", err)
	}

	return fn(db)
}

// initSchema creates the necessary tables if they don't exist
func initSchema(db *sql.DB) error {
	_, err := db.Exec(`
		CREATE TABLE IF NOT EXISTS prompts (
			file_path TEXT PRIMARY KEY,
			prompt TEXT NOT NULL,
			workflow TEXT NOT NULL
		)
	`)
	return err
}
