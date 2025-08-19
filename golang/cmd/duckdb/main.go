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

func LoadConfig(path string) (*koanf.Koanf, error) {
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

	k, err := LoadConfig("../config.yml")
	if err != nil {
		return err
	}

	db, err := sqlx.Open("duckdb", k.String("db_path"))
	if err != nil {
		return err
	}
	defer db.Close()
	logger.Info("connected to database", "db_path", k.String("db_path"))

	minScore := k.Int("min_score")
	taggedPosts, err := FindTagString(ctx, db, FindPostsOptions{
		Tag:      k.String("tag"),
		MinScore: &minScore,
		Random:   true,
	})
	if err != nil {
		return fmt.Errorf("find tag string: %w", err)
	}
	logger.Info("found tagged posts", "posts", taggedPosts)

	return nil
}

func DBDump(ctx context.Context, db *sqlx.DB, logger slog.Logger, k *koanf.Koanf) error {
	if err := PrintTableInfo(ctx, db, "tag_counts"); err != nil {
		return fmt.Errorf("print table info: %w", err)
	}

	if err := PrintTableInfo(ctx, db, "post_tags"); err != nil {
		return fmt.Errorf("print table info: %w", err)
	}

	var tables []string
	if err := db.SelectContext(ctx, &tables, "SHOW TABLES;"); err != nil {
		return fmt.Errorf("show tables: %w", err)
	}
	logger.Info("tables in database", "tables", tables)
	return nil
}
