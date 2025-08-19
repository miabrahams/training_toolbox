package main

import (
	"context"
	"fmt"
	"log/slog"
	"os"

	"github.com/jmoiron/sqlx"

	"github.com/knadh/koanf/parsers/yaml"
	"github.com/knadh/koanf/providers/file"
	"github.com/knadh/koanf/v2"
	_ "github.com/marcboeker/go-duckdb"
)

func load_config(path string) (*koanf.Koanf, error) {
	k := koanf.New(".")
	err := k.Load(file.Provider(path), yaml.Parser())
	return k, err
}

func main() {
	if err := run_main(); err != nil {
		slog.Error(err.Error())
		os.Exit(1)
	}
}

func run_main() error {
	ctx := context.Background()
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelInfo}))
	slog.SetDefault(logger)

	k, err := load_config("../config.yml")
	if err != nil {
		return err
	}

	db, err := sqlx.Open("duckdb", k.String("db_path"))
	if err != nil {
		return err
	}
	defer db.Close()

	taggedPost, err := FindTagString(ctx, db, k.String("tag"), k.Int("min_score"))
	if err != nil {
		return fmt.Errorf("find tag string: %w", err)
	}
	logger.Info("found tagged post", "post", taggedPost)

	/*
		logger.Info("connected to database", "db_path", k.String("db_path"))
		if err := print_table_info(ctx, db, "tag_counts"); err != nil {
			return fmt.Errorf("print table info: %w", err)
		}

		if err := print_table_info(ctx, db, "post_tags"); err != nil {
			return fmt.Errorf("print table info: %w", err)
		}
	*/

	var tables []string
	if err := db.SelectContext(ctx, &tables, "SHOW TABLES;"); err != nil {
		return fmt.Errorf("show tables: %w", err)
	}
	logger.Info("tables in database", "tables", tables)

	/*
		var posts []Post
		if err := db.SelectContext(ctx, &posts, "SELECT * FROM (select * from posts ORDER BY RANDOM()) LIMIT 2"); err != nil {
			return err
		}

		for _, post := range posts {
			logger.Info("row", "value", post)
		}
		if err := findPostsWithTag(ctx, db, k.String("tag"), k.Int("min_score")); err != nil {
			return fmt.Errorf("find posts with tag: %w", err)
		}
	*/

	return nil
}
