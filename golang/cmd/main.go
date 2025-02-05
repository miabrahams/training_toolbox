package main

import (
	"flag"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"training_toolbox/internal/parser"
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

	if flag.NFlag() == 0 || flag.NFlag() > 1 {
		flag.Usage()
		return fmt.Errorf("missing file or directory")
	}

	if dir != nil {
		return parseDirectory(*dir)
	}

	if file != nil {
		return parseFileCommand(*file)
	}

	flag.Usage()
	return nil
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

func parseDirectory(dir string) error {
	return filepath.WalkDir(dir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			return nil
		}
		// Process only .png files.
		if strings.HasSuffix(strings.ToLower(d.Name()), ".png") {
			fmt.Printf("Processing %s\n", path)
			// addFileToSqlite is unimplemented.
			if err := addFileToSqlite(path); err != nil {
				fmt.Fprintf(os.Stderr, "Error processing %s: %v\n", path, err)
			}
		}
		return nil
	})
}

func addFileToSqlite(filepath string) error {
	fmt.Printf("parseFile2 not implemented for %s\n", filepath)
	return nil
}
