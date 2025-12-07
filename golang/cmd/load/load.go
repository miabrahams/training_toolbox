package main

import (
	"flag"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"

	"training_toolbox/internal/config"
	promptdb "training_toolbox/internal/database/prompts"
	"training_toolbox/internal/parser"

	_ "github.com/mattn/go-sqlite3"

	_ "github.com/marcboeker/go-duckdb"

	"database/sql"
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

// promptDB abstracts DB differences (sqlite vs duckdb).
type promptDB interface {
	GetExistingFilePaths() (map[string]struct{}, error)
	InsertBatch(batch []fileResult) error
	Close() error
}

func run() error {
	fromConfig := flag.String("config", "", "Path to config file")
	file := flag.String("file", "", "Path to a PNG file")
	dir := flag.String("dir", "", "Path to a directory containing PNG files")
	dbpath := flag.String("db", "", "Path to a sqlite or duckdb database (use .sqlite/.db for SQLite, .duckdb for DuckDB)")
	flag.Parse()

	if fromConfig != nil && *fromConfig != "" {
		cfg, err := config.LoadConfig(*fromConfig)
		if err != nil {
			return fmt.Errorf("load config: %w", err)
		}
		return parseDirectoriesFromConfig(cfg, dbpath)
	}

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

func parseDirectoriesFromConfig(cfg config.Config, dbpath *string) error {
	paths := cfg.PromptExtractPaths()
	if len(paths) == 0 {
		return fmt.Errorf("no directories specified in config for prompt extraction")
	}
	for _, dir := range paths {
		if err := parseDirectoryCommand(dir, dbpath); err != nil {
			return err
		}
	}
	return nil
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
	promptdb.FilePrompt
	err error
}

func parseDirectoryCommand(root string, dbpath *string) error {
	paths, err := getPngPaths(root)
	if err != nil {
		return fmt.Errorf("error getting PNG paths: %w", err)
	}

	var dbPath string
	if dbpath == nil || *dbpath == "" {
		// default to sqlite if not provided
		// TODO: use config default
		dbPath = filepath.Join(root, "prompts.sqlite")
	} else {
		dbPath = *dbpath
	}

	pdb, err := openPromptDB(dbPath)
	if err != nil {
		return fmt.Errorf("open db: %w", err)
	}
	defer pdb.Close()

	return parseDirectory(paths, pdb)
}

// parseDirectory now uses the promptDB interface for DB ops
func parseDirectory(paths []string, pdb promptDB) error {
	existingPaths, err := pdb.GetExistingFilePaths()
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
	filesCh := make(chan string, numWorkers) // TODO: Mutex
	resultsCh := make(chan fileResult)

	var wg sync.WaitGroup
	wg.Add(numWorkers)
	worker := func() {
		defer wg.Done()
		for path := range filesCh {
			prompt, workflow, err := parser.ExtractFileChunks(path)
			fileprompt := promptdb.FilePrompt{FilePath: path, Prompt: prompt, Workflow: workflow}
			resultsCh <- fileResult{FilePrompt: fileprompt, err: err}
		}
	}
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

	processed := 0
	batch := make([]fileResult, 0, batchSize)
	for res := range resultsCh {
		processed++
		if res.err != nil {
			fmt.Fprintf(os.Stderr, "\n\nerror processing %s: %v\n\n", res.FilePath, res.err)
		}

		batch = append(batch, res)
		if len(batch) >= batchSize {
			if err := pdb.InsertBatch(batch); err != nil {
				fmt.Fprintf(os.Stderr, "\n\nfailed to insert batch into db: %v\n\n", err)
			}
			batch = batch[:0]
		}
		fmt.Printf("\rProcessed %d/%d new files", processed, len(filesToProcess))
	}

	if len(batch) > 0 {
		if err := pdb.InsertBatch(batch); err != nil {
			fmt.Fprintf(os.Stderr, "\n\nfailed to insert batch into db: %v\n\n", err)
		}
	}

	fmt.Println("\nDone!")
	return nil
}

type basePromptDB struct {
	db *sql.DB
}

// common schema creation for both drivers
func (b *basePromptDB) ensureSchema() error {
	_, err := b.db.Exec(`
		CREATE TABLE IF NOT EXISTS prompts (
			file_path TEXT PRIMARY KEY,
			prompt    TEXT,
			workflow  TEXT
		)
	`)
	return err
}

type sqlitePromptDB struct {
	basePromptDB
}

type duckdbPromptDB struct {
	basePromptDB
}

func (s *sqlitePromptDB) GetExistingFilePaths() (map[string]struct{}, error) {
	rows, err := s.db.Query("SELECT file_path FROM prompts")
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	existing := make(map[string]struct{})
	for rows.Next() {
		var p string
		if err := rows.Scan(&p); err != nil {
			return nil, err
		}
		existing[p] = struct{}{}
	}
	return existing, rows.Err()
}

func (d *duckdbPromptDB) GetExistingFilePaths() (map[string]struct{}, error) {
	// same as sqlite
	return (&sqlitePromptDB{basePromptDB: d.basePromptDB}).GetExistingFilePaths()
}

func buildUpsertStatement(num int) (string, []any) {
	valueStrings := make([]string, 0, num)
	valueArgs := make([]any, 0, num*3)
	for range num {
		valueStrings = append(valueStrings, "(?, ?, ?)")
	}
	stmt := fmt.Sprintf(
		"INSERT INTO prompts (file_path, prompt, workflow) VALUES %s ON CONFLICT(file_path) DO UPDATE SET prompt=excluded.prompt, workflow=excluded.workflow",
		strings.Join(valueStrings, ","),
	)
	return stmt, valueArgs
}

func (s *sqlitePromptDB) InsertBatch(batch []fileResult) error {
	if len(batch) == 0 {
		return nil
	}
	stmt, args := buildUpsertStatement(len(batch))
	for _, item := range batch {
		args = append(args, item.FilePath, item.Prompt, item.Workflow)
	}
	_, err := s.db.Exec(stmt, args...)
	return err
}

func (d *duckdbPromptDB) InsertBatch(batch []fileResult) error {
	return (&sqlitePromptDB{basePromptDB: d.basePromptDB}).InsertBatch(batch)
}

func (b *basePromptDB) Close() error {
	return b.db.Close()
}

func openPromptDB(dbPath string) (promptDB, error) {
	ext := strings.ToLower(filepath.Ext(dbPath))
	switch ext {
	case ".duckdb":
		db, err := sql.Open("duckdb", dbPath)
		if err != nil {
			return nil, err
		}
		p := &duckdbPromptDB{basePromptDB{db: db}}
		if err := p.ensureSchema(); err != nil {
			_ = db.Close()
			return nil, err
		}
		return p, nil
	default:
		// default to sqlite
		dsn := fmt.Sprintf("file:%s?_busy_timeout=5000&_fk=1", dbPath)
		db, err := sql.Open("sqlite3", dsn)
		if err != nil {
			return nil, err
		}
		p := &sqlitePromptDB{basePromptDB{db: db}}
		if err := p.ensureSchema(); err != nil {
			_ = db.Close()
			return nil, err
		}
		return p, nil
	}
}
