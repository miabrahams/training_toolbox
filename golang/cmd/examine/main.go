package main

import (
	"context"
	"database/sql"
	"encoding/json"
	"flag"
	"fmt"
	"os"

	"training_toolbox/internal/database"
	prompts "training_toolbox/internal/database/sqlc"
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
	dbpath := flag.String("db", "", "Path to a sqlite database")
	flag.Parse()

	if *dbpath == "" {
		flag.Usage()
		return fmt.Errorf("missing database")
	}
	return examineDatabase(*dbpath)
}

func examineDatabase(dbpath string) error {
	// WithDB creates or opens the database and sets up the table.
	return database.WithDB(dbpath, func(db *sql.DB) error {
		// Create a new sqlc queries instance.
		q := prompts.New(db)
		// ListPrompts is a sqlc-generated method returning []FilePrompt.
		tp, err := q.GetTestPrompt(context.Background())
		if err != nil {
			return fmt.Errorf("error listing prompts: %w", err)
		}
		// Parse the prompt JSON.
		parsedPrompt, err := parser.ParseChunk(tp.Prompt)
		if err != nil {
			return fmt.Errorf("error parsing prompt for %s: %w", tp.FilePath, err)
		}
		// Parse the workflow JSON.
		parsedWorkflow, err := parser.ParseChunk(tp.Workflow)
		if err != nil {
			return fmt.Errorf("error parsing workflow for %s: %w", tp.FilePath, err)
		}
		// Pretty-print the parsed JSON.
		promptJSON, _ := json.MarshalIndent(parsedPrompt, "", "  ")
		workflowJSON, _ := json.MarshalIndent(parsedWorkflow, "", "  ")
		fmt.Printf("File: %s\n", tp.FilePath)
		fmt.Printf("Prompt:\n%s\n", promptJSON)
		fmt.Printf("Workflow:\n%s\n\n", workflowJSON)
		return nil
	})
}
