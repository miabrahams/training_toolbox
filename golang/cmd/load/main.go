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

	"training_toolbox/internal/database"
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
	dbpath := flag.String("db", "", "Path to a sqlite database")
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
	return parseDirectoryCommand(*dir, dbpath)
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
		return nil, fmt.Errorf("no .png files found in %s", root)
	}
	return paths, nil
}

type fileResult struct {
	database.FilePrompt
	err error
}

func parseDirectoryCommand(root string, dbpath *string) error {
	paths, err := getPngPaths(root)
	if err != nil {
		return fmt.Errorf("error getting PNG paths: %w", err)
	}

	var dbPath string
	if dbpath == nil {
		dbPath = filepath.Join(root, "prompts.sqlite")
	} else {
		dbPath = *dbpath
	}

	return database.WithDB(dbPath, func(db *sql.DB) error {
		return parseDirectory(paths, db)
	})
}

// getExistingFilePaths retrieves all file paths that are already in the database
func getExistingFilePaths(db *sql.DB) (map[string]struct{}, error) {
	rows, err := db.Query("SELECT file_path FROM prompts")
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	existingPaths := make(map[string]struct{})
	for rows.Next() {
		var path string
		if err := rows.Scan(&path); err != nil {
			return nil, err
		}
		existingPaths[path] = struct{}{}
	}

	if err := rows.Err(); err != nil {
		return nil, err
	}

	return existingPaths, nil
}

func parseDirectory(paths []string, db *sql.DB) error {
	existingPaths, err := getExistingFilePaths(db)
	if err != nil {
		return fmt.Errorf("error retrieving existing files: %w", err)
	}

	// Filter out files that are already in the database
	var filesToProcess []string
	for _, path := range paths {
		if _, exists := existingPaths[path]; !exists {
			filesToProcess = append(filesToProcess, path)
		}
	}

	skipped := len(paths) - len(filesToProcess)
	fmt.Printf("Found %d files, skipping %d already loaded files, processing %d new files\n",
		len(paths), skipped, len(filesToProcess))

	if len(filesToProcess) == 0 {
		fmt.Println("All files are already loaded in the database.")
		return nil
	}

	numWorkers := runtime.NumCPU()
	filesCh := make(chan string)
	resultsCh := make(chan fileResult)

	var wg sync.WaitGroup
	worker := func() {
		defer wg.Done()
		for path := range filesCh {
			prompt, workflow, err := parser.ExtractFileChunks(path)
			fileprompt := database.FilePrompt{Path: path, Prompt: prompt, Workflow: workflow}
			resultsCh <- fileResult{FilePrompt: fileprompt, err: err}
		}
	}
	wg.Add(numWorkers)
	for range numWorkers {
		go worker()
	}

	go func() {
		for _, p := range filesToProcess {
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
			valueArgs = append(valueArgs, item.Path)
			valueArgs = append(valueArgs, item.Prompt)
			valueArgs = append(valueArgs, item.Workflow)
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
			fmt.Fprintf(os.Stderr, "\n\nerror processing %s: %v\n\n", res.Path, res.err)
		}

		batch = append(batch, res)
		if len(batch) >= batchSize {
			if err := insertBatch(batch); err != nil {
				fmt.Fprintf(os.Stderr, "\n\nfailed to insert batch into db: %v\n\n", err)
			}
			batch = batch[:0]
		}
		// Update progress
		fmt.Printf("\rProcessed %d/%d new files", processed, len(filesToProcess))
	}

	if len(batch) > 0 {
		if err := insertBatch(batch); err != nil {
			fmt.Fprintf(os.Stderr, "\n\nfailed to insert batch into db: %v\n\n", err)
		}
	}

	fmt.Println("\nDone.")
	return nil

}
