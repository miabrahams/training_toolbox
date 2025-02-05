package main

import (
	"database/sql"
	"flag"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"

	"training_toolbox/internal/parser"

	_ "github.com/mattn/go-sqlite3"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "%s\n", err)
		os.Exit(1)
	}
}

const (
	batchSize = 25
)

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
	return parseDirectoryCommand(*dir)
}

func parseFileCommand(file string) error {
	prompt, workflow, err := parser.ExtractFileChunks(file)
	if err != nil {
		return fmt.Errorf("error parsing file: %w", err)
	}

	fmt.Printf("Prompt: %s\n", prompt)
	fmt.Printf("Workflow: %s\n", workflow)
	return nil
}

func getPngPaths(root string) ([]string, error) {
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
		return nil, err
	}

	total := len(paths)
	if total == 0 {
		return nil, fmt.Errorf("No PNG files found.")
	}
	return paths, nil
}

type fileResult struct {
	path     string
	prompt   string
	workflow string
	err      error
}

func parseDirectoryCommand(root string) error {

	paths, err := getPngPaths(root)
	if err != nil {
		return fmt.Errorf("error getting PNG paths: %w", err)
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

	return parseDirectory(paths, db)
}

func parseDirectory(paths []string, db *sql.DB) error {

	numWorkers := runtime.NumCPU()
	filesCh := make(chan string)
	resultsCh := make(chan fileResult)

	var wg sync.WaitGroup
	worker := func() {
		defer wg.Done()
		for path := range filesCh {
			prompt, workflow, err := parser.ExtractFileChunks(path)
			resultsCh <- fileResult{path, prompt, workflow, err}
		}
	}
	wg.Add(numWorkers)
	for i := 0; i < numWorkers; i++ {
		go worker()
	}

	go func() {
		for _, p := range paths {
			filesCh <- p
		}
		close(filesCh)
	}()

	go func() {
		wg.Wait()
		close(resultsCh)
	}()

	insertBatch := func(batch []fileResult) error {
		valueStrings := make([]string, 0, len(batch))
		valueArgs := make([]interface{}, 0, len(batch)*3)
		for _, item := range batch {
			valueStrings = append(valueStrings, "(?, ?, ?)")
			valueArgs = append(valueArgs, item.path)
			valueArgs = append(valueArgs, item.prompt)
			valueArgs = append(valueArgs, item.workflow)
		}
		stmt := fmt.Sprintf("INSERT OR REPLACE INTO prompts (file_path, prompt, workflow) VALUES %s", strings.Join(valueStrings, ","))
		_, err := db.Exec(stmt, valueArgs...)
		return err
	}

	processed := 0
	batch := make([]fileResult, 0, batchSize)
	for res := range resultsCh {
		processed++
		if res.err != nil {
			fmt.Fprintf(os.Stderr, "\n\nerror processing %s: %v\n\n", res.path, res.err)
		}

		batch = append(batch, res)
		if len(batch) >= batchSize {
			if err := insertBatch(batch); err != nil {
				fmt.Fprintf(os.Stderr, "\n\nfailed to insert batch into db: %v\n\n", err)
			}
			batch = batch[:0]
		}
		// Update progress
		fmt.Printf("\rProcessed %d/%d files", processed, len(paths))
	}

	if len(batch) > 0 {
		if err := insertBatch(batch); err != nil {
			fmt.Fprintf(os.Stderr, "\n\nfailed to insert batch into db: %v\n\n", err)
		}
	}

	fmt.Println("\nDone.")
	return nil

}
