package main

import (
	"database/sql"
	"flag"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"

	"training_toolbox/internal/parser"

	_ "github.com/mattn/go-sqlite3"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "%s\n", err)
		os.Exit(1)
	}
}

func run() error {
	file := flag.String("file", "", "Path to a PNG file")
	dir := flag.String("dir", "", "Path to a directory containing PNG files")
	flag.Parse()

	if *file == "" && *dir == "" {
		flag.Usage()
		return fmt.Errorf("missing file or directory")
	}
	if *file != "" && *dir != "" {
		flag.Usage()
		return fmt.Errorf("please provide either a file or directory, not both")
	}

	if *file != "" {
		return parseFileCommand(*file)
	}
	return parseDirectory(*dir)
}

func parseFileCommand(file string) error {
	prompt, workflow, err := parser.ParseFile(file)
	if err != nil {
		return fmt.Errorf("error parsing file: %w", err)
	}

	fmt.Printf("Prompt: %s\n", prompt)
	fmt.Printf("Workflow: %s\n", workflow)
	return nil
}

func parseDirectory(root string) error {
	// First pass: collect all PNG file paths.
	var paths []string
	err := filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			return nil
		}
		if strings.HasSuffix(strings.ToLower(d.Name()), ".png") {
			paths = append(paths, path)
		}
		return nil
	})
	if err != nil {
		return err
	}

	total := len(paths)
	if total == 0 {
		fmt.Println("No PNG files found.")
		return nil
	}

	// Create (or open) the sqlite DB at the root.
	dbPath := filepath.Join(root, "prompts.sqlite")
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

	// Second pass: process each file.
	processed := 0
	for _, path := range paths {
		prompt, workflow, err := parser.ParseFile(path)
		processed++
		if err != nil {
			fmt.Fprintf(os.Stderr, "\nError processing %s: %v\n", path, err)
			continue
		}

		// Upsert into the sqlite DB.
		if _, err := db.Exec(
			"REPLACE INTO prompts(file_path, prompt, workflow) VALUES(?, ?, ?)",
			path, prompt, workflow); err != nil {
			fmt.Fprintf(os.Stderr, "\nDB insertion error for %s: %v\n", path, err)
			continue
		}

		// Display progress.
		fmt.Printf("\rProcessed %d/%d files", processed, total)
	}
	fmt.Println("\nProcessing complete.")
	return nil
}
