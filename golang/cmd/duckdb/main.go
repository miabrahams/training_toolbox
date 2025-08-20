package main

import (
	"context"
	"fmt"
	"log/slog"
	"math/rand/v2"
	"os"
	"strings"
	"time"

	"github.com/jmoiron/sqlx"

	"github.com/knadh/koanf/parsers/yaml"
	"github.com/knadh/koanf/providers/confmap"
	"github.com/knadh/koanf/providers/file"
	"github.com/knadh/koanf/v2"
	_ "github.com/marcboeker/go-duckdb"
)

func LoadConfig(path string) (*koanf.Koanf, error) {
	k := koanf.New(".")

	// defaults
	k.Load(confmap.Provider(map[string]any{
		"comfy.url":               "http://localhost:8188",
		"generations.batch_count": 2,
		"generations.pause_time":  time.Second * 5,
	}, "."), nil)

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

	// DuckDB creates a new database if file doesn't exist
	dbPath := k.String("db.path")
	if _, err := os.Stat(dbPath); err != nil {
		return fmt.Errorf("error locating database file %s: %w", k.String("db.path"), err)
	}

	db, err := sqlx.Open("duckdb", dbPath+"?access_mode=read_only")
	if err != nil {
		return err
	}
	defer db.Close()
	logger.Info("connected to database", "path", k.String("db.path"))

	minScore := k.Int("min_score")
	taggedPosts, err := FindTagString(ctx, db, FindPostsOptions{
		Tag:      k.String("tag"),
		MinScore: &minScore,
		Random:   true,
	})
	if err != nil {
		return fmt.Errorf("find tag string: %w", err)
	}

	comfyURL := k.String("comfy.url")
	batchCount := k.Int("generations.batch_count")
	pauseTime := k.Duration("generations.pause_time")

	client := ComfyAPIClient{
		BaseURL: comfyURL,
	}
	client.Init()

	for i, post := range taggedPosts {
		logger.Info("processing post", "index", i, "post", post.ID)
		rand.Shuffle(len(post.Tags), func(i int, j int) {
			post.Tags[i], post.Tags[j] = post.Tags[j], post.Tags[i]
		})

		// TODO: see about this type conversion
		strTags := make([]string, 0, len(post.Tags))
		for _, i := range post.Tags {
			if tag, ok := i.(string); ok {
				strTags = append(strTags, tag)
			} else {
				return fmt.Errorf("tag is not a string %T", tag)
			}
		}
		prompt := strings.Join(strTags, ", ")

		logger.Info("sending promptReplace", "prompt", prompt, "width", post.ImageWidth, "height", post.ImageHeight)
		if err := client.SendPromptReplace(ctx, prompt, post.ImageWidth, post.ImageHeight); err != nil {
			logger.Error("promptReplace failed", "error", err)
			continue
		}
		time.Sleep(pauseTime / 2.0)

		logger.Info("sending generateImages", "count", batchCount)
		if err := client.SendGenerate(ctx, batchCount); err != nil {
			logger.Error("sendGenerateImages failed", "error", err)
			continue
		}

		if i < len(taggedPosts)-1 {
			logger.Info("waiting for next post generation")
			time.Sleep(pauseTime)
		}
	}
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
