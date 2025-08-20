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
		"db.debug":                false,
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
	level := slog.LevelVar{}
	logger := slog.New(
		slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: &level}),
	)
	slog.SetDefault(logger)

	k, err := LoadConfig("../config.yml")
	if err != nil {
		return err
	}

	if k.String("log.level") != "" {
		levelReader := slog.Level(0)
		err := levelReader.UnmarshalText([]byte(k.String("log.level")))
		if err != nil {
			return fmt.Errorf("invalid log level: %w", err)
		}
		level.Set(levelReader)
	}

	db, err := loadDB(k)
	if err != nil {
		return fmt.Errorf("load db: %w", err)
	}
	defer db.Close()
	slog.Info("connected to database", "path", k.String("db.path"))

	if k.Bool("db.debug") {
		DBDump(ctx, db)
	}

	posts, err := loadPosts(ctx, k, db)
	if err != nil {
		return fmt.Errorf("load posts: %w", err)
	}

	if err := sendPosts(ctx, k, posts); err != nil {
		return fmt.Errorf("send posts: %w", err)
	}
	return nil
}

func loadDB(k *koanf.Koanf) (*sqlx.DB, error) {
	// DuckDB creates a new database if file doesn't exist
	dbPath := k.String("db.path")
	if _, err := os.Stat(dbPath); err != nil {
		return nil, fmt.Errorf("error locating database file %s: %w", k.String("db.path"), err)
	}

	return sqlx.Open("duckdb", dbPath+"?access_mode=read_only")
}

func loadPosts(ctx context.Context, k *koanf.Koanf, db *sqlx.DB) ([]TaggedPost, error) {
	minScore := k.Int("min_score")
	taggedPosts, err := FindTagString(ctx, db, FindPostsOptions{
		Tag:      k.String("tag"),
		MinScore: &minScore,
		Random:   true,
	})
	if err != nil {
		return nil, fmt.Errorf("find tag string: %w", err)
	}
	return taggedPosts, nil
}

func sendPosts(ctx context.Context, k *koanf.Koanf, posts []TaggedPost) error {
	comfyURL := k.String("comfy.url")
	batchCount := k.Int("generations.batch_count")
	pauseTime := k.Duration("generations.pause_time")

	client := ComfyAPIClient{
		BaseURL: comfyURL,
	}
	client.Init()

	for i, post := range posts {
		slog.Info("processing post", "index", i, "post", post.ID)
		rand.Shuffle(len(post.Tags), func(i int, j int) {
			post.Tags[i], post.Tags[j] = post.Tags[j], post.Tags[i]
		})

		// TODO: see about this type conversion
		/*
			strTags := make([]string, 0, len(post.Tags))
			for _, i := range post.Tags {
				if tag, ok := i.(string); ok {
					strTags = append(strTags, tag)
				} else {
					return fmt.Errorf("tag is not a string %T", tag)
				}
			}
		*/
		prompt := strings.Join(post.Tags, ", ")

		slog.Info("sending promptReplace", "prompt", prompt, "width", post.ImageWidth, "height", post.ImageHeight)
		if err := client.SendPromptReplace(ctx, prompt, post.ImageWidth, post.ImageHeight); err != nil {
			slog.Error("promptReplace failed", "error", err)
			continue
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(pauseTime / 2.0):
		}

		slog.Info("sending generateImages", "count", batchCount)
		if err := client.SendGenerate(ctx, batchCount); err != nil {
			slog.Error("sendGenerateImages failed", "error", err)
			continue
		}

		if i < len(posts)-1 {
			slog.Info("waiting...")
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(pauseTime):
			}
		}
	}
	return nil
}
