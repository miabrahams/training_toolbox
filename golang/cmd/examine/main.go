package main

import (
	"context"
	"database/sql"
	"encoding/json"
	"flag"
	"fmt"
	"os"

	promptdb "training_toolbox/internal/database/prompts"
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
	return promptdb.WithDB(dbpath, func(db *sql.DB) error {
		q := promptdb.New(db)
		tp, err := q.GetTestPrompt(context.Background())
		if err != nil {
			return fmt.Errorf("error listing prompts: %w", err)
		}
		parsedPrompt, err := parser.ParseChunk(tp.Prompt)
		if err != nil {
			return fmt.Errorf("error parsing prompt for %s: %w", tp.FilePath, err)
		}
		parsedWorkflow, err := parser.ParseChunk(tp.Workflow)
		if err != nil {
			return fmt.Errorf("error parsing workflow for %s: %w", tp.FilePath, err)
		}
		promptJSON, _ := json.MarshalIndent(parsedPrompt, "", "  ")
		workflowJSON, _ := json.MarshalIndent(parsedWorkflow, "", "  ")
		fmt.Printf("File: %s\n", tp.FilePath)
		fmt.Printf("Prompt:\n%s\n", promptJSON)
		fmt.Printf("Workflow:\n%s\n\n", workflowJSON)
		return nil
	})
}
