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

const (
	comfyUrlConfig      = "comfy.url"
	genBatchCountConfig = "generations.batch_count"
	genPauseTimeConfig  = "generations.pause_time"
	prefixConfig        = "generations.prefix"
	genStripTagsConfig  = "search.strip_tags"
	dbDebugConfig       = "db.debug"
	dbPathConfig        = "db.path"
	tagsConfig          = "search.tags"
	excludeTagsConfig   = "search.exclude_tags"
	minScoreConfig      = "search.min_score"
	limitConfig         = "search.limit"
	logLevelConfig      = "log.level"
)

func LoadConfig(path string) (*koanf.Koanf, error) {
	k := koanf.New(".")

	defaults := map[string]any{
		comfyUrlConfig:      "http://localhost:8188",
		genBatchCountConfig: 2,
		genPauseTimeConfig:  time.Second * 5,
		dbDebugConfig:       false,
	}

	k.Load(confmap.Provider(defaults, "."), nil)

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

	if k.String(logLevelConfig) != "" {
		levelReader := slog.Level(0)
		err := levelReader.UnmarshalText([]byte(k.String(logLevelConfig)))
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
	slog.Info("connected to database", "path", k.String(dbPathConfig))

	if k.Bool(dbDebugConfig) {
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
	dbPath := k.String(dbPathConfig)
	if _, err := os.Stat(dbPath); err != nil {
		return nil, fmt.Errorf("error locating database file %s: %w", k.String(dbPathConfig), err)
	}

	return sqlx.Open("duckdb", dbPath+"?access_mode=read_only")
}

func loadPosts(ctx context.Context, k *koanf.Koanf, db *sqlx.DB) ([]TaggedPost, error) {
	minScore := k.Int(minScoreConfig)
	taggedPosts, err := FindPostsWithAllTags(ctx, db, FindPostsOptions{
		Tags:        k.Strings(tagsConfig),
		ExcludeTags: k.Strings(excludeTagsConfig),
		MinScore:    &minScore,
		Random:      true,
		Limit:       k.Int(limitConfig),
	})
	if err != nil {
		return nil, fmt.Errorf("find tag string: %w", err)
	}
	return taggedPosts, nil
}

func tagsToPrompt(b *strings.Builder, r *strings.Replacer, tags []string, stripTags map[string]struct{}) {
	rand.Shuffle(len(tags), func(i int, j int) {
		tags[i], tags[j] = tags[j], tags[i]
	})

	wroteTag := false
	for i, tag := range tags {
		if (i > 0) && wroteTag {
			b.WriteString(", ")
		}
		if _, ok := stripTags[tag]; ok {
			continue
		}
		tag = strings.ReplaceAll(tag, "_", " ")
		b.WriteString(r.Replace(tag))
		wroteTag = true
	}
}

func sendPosts(ctx context.Context, k *koanf.Koanf, posts []TaggedPost) error {
	comfyURL := k.String(comfyUrlConfig)
	batchCount := k.Int(genBatchCountConfig)
	pauseTime := k.Duration(genPauseTimeConfig)

	client := ComfyAPIClient{
		BaseURL: comfyURL,
	}
	client.Init()

	b := strings.Builder{}
	r := strings.NewReplacer("_", " ", "(", "\\(", ")", "\\)")
	stripTags := make(map[string]struct{}, len(k.Strings(genStripTagsConfig)))
	for _, tag := range k.Strings(genStripTagsConfig) {
		stripTags[tag] = struct{}{}
	}

	for i, post := range posts {
		slog.Info("processing post", "index", i, "post", post.ID)

		// Build prompt
		b.Grow(2048)
		if prefix := k.String(prefixConfig); prefix != "" {
			b.WriteString(prefix)
			b.WriteString(", ")
		}
		tagsToPrompt(&b, r, post.Tags, stripTags)
		if postfix := k.String("generations.postfix"); postfix != "" {
			b.WriteString(", ")
			b.WriteString(postfix)
		}
		prompt := b.String()
		b.Reset()

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
