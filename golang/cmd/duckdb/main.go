package main

import (
	"context"
	"fmt"
	"log/slog"
	"math/rand/v2"
	"os"
	"slices"
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
	comfyUrlKey       = "comfy.url"
	genBatchCountKey  = "generations.batch_count"
	genPauseTimeKey   = "generations.pause_time"
	prefixKey         = "generations.prefix"
	genStripTagsKey   = "search.strip_tags"
	dbDebugKey        = "db.debug"
	dbPathKey         = "db.path"
	searchTagsKey     = "search.tags"
	excludeTagsKey    = "search.exclude_tags"
	minScoreKey       = "search.min_score"
	minFavsKey        = "search.min_favs"
	limitKey          = "search.limit"
	logLevelKey       = "log.level"
	styleKey          = "generations.style"
	styleConfigsKey   = "styleConfigs"
	defaultPrefixKey  = "generations.default_prefix"
	defaultPostfixKey = "generations.default_postfix"
)

func LoadConfig(path string) (*koanf.Koanf, error) {
	k := koanf.New(".")

	defaults := map[string]any{
		comfyUrlKey:      "http://localhost:8188",
		genBatchCountKey: 2,
		genPauseTimeKey:  time.Second * 5,
		dbDebugKey:       false,
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

	if k.String(logLevelKey) != "" {
		levelReader := slog.Level(0)
		err := levelReader.UnmarshalText([]byte(k.String(logLevelKey)))
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
	slog.Info("connected to database", "path", k.String(dbPathKey))

	if k.Bool(dbDebugKey) {
		DBDump(ctx, db)
	}

	posts, err := loadPosts(ctx, k, db)
	if err != nil {
		return fmt.Errorf("load posts: %w", err)
	}

	genConfig := buildGenerationConfig(k)
	client := ComfyAPIClient{
		BaseURL: k.String(comfyUrlKey),
	}
	client.Init()

	if err := sendPosts(ctx, client, genConfig, posts); err != nil {
		return fmt.Errorf("send posts: %w", err)
	}
	return nil
}

func loadDB(k *koanf.Koanf) (*sqlx.DB, error) {
	// DuckDB creates a new database if file doesn't exist
	dbPath := k.String(dbPathKey)
	if _, err := os.Stat(dbPath); err != nil {
		return nil, fmt.Errorf("error locating database file %s: %w", k.String(dbPathKey), err)
	}

	return sqlx.Open("duckdb", dbPath+"?access_mode=read_only")
}

func loadPosts(ctx context.Context, k *koanf.Koanf, db *sqlx.DB) ([]TaggedPost, error) {
	var minScore *int
	if k.Exists(minScoreKey) {
		val := k.Int(minScoreKey)
		minScore = &val
	}
	var minFavs *int
	if k.Exists(minFavsKey) {
		val := k.Int(minFavsKey)
		minFavs = &val
	}
	taggedPosts, err := FindPostsWithAllTags(ctx, db, FindPostsOptions{
		Tags:        k.Strings(searchTagsKey),
		ExcludeTags: k.Strings(excludeTagsKey),
		MinScore:    minScore,
		MinFavs:     minFavs,
		Random:      true,
		Limit:       k.Int(limitKey),
	})
	if err != nil {
		return nil, fmt.Errorf("find tag string: %w", err)
	}
	return taggedPosts, nil
}

func tagsToPrompt(b *strings.Builder, r *strings.Replacer, post TaggedPost, stripTags map[string]struct{}) {
	tags := slices.Clone(post.Tags)
	tagcats := slices.Clone(post.TagCategory)
	rand.Shuffle(len(tags), func(i int, j int) {
		tags[i], tags[j] = tags[j], tags[i]
		tagcats[i], tagcats[j] = tagcats[j], tagcats[i]
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

type GenerationConfig struct {
	BatchCount int
	PauseTime  time.Duration
	Prefix     string
	Postfix    string
	StripTags  []string
}

const bufferLength = 4096

func buildGenerationConfig(k *koanf.Koanf) GenerationConfig {

	// start with defaults
	config := GenerationConfig{
		BatchCount: k.Int(genBatchCountKey),
		PauseTime:  k.Duration(genPauseTimeKey),
		Prefix:     k.String(defaultPrefixKey),
		Postfix:    k.String(defaultPostfixKey),
		StripTags:  k.Strings(genStripTagsKey),
	}

	// merge style-specific config if available
	style := k.String(styleKey)
	if style != "" {
		styleConfigKey := fmt.Sprintf("%s.%s", styleConfigsKey, style)
		if k.Exists(styleConfigKey) {
			if prefix := k.String(styleConfigKey + ".prefix"); prefix != "" {
				config.Prefix = prefix
			}
			if postfix := k.String(styleConfigKey + ".postfix"); postfix != "" {
				config.Postfix = postfix
			}
			stripTags := k.Strings(styleConfigKey + ".strip_tags")
			config.StripTags = append(config.StripTags, stripTags...)
		}
	}

	return config
}

func sendPosts(ctx context.Context, client ComfyAPIClient, config GenerationConfig, posts []TaggedPost) error {
	b := strings.Builder{}
	r := strings.NewReplacer("_", " ", "(", "\\(", ")", "\\)")
	stripTags := make(map[string]struct{}, len(config.StripTags))
	for _, tag := range config.StripTags {
		stripTags[tag] = struct{}{}
	}

	numTooLong := 0

	for i, post := range posts {
		slog.Info("processing post", "index", i, "post", post.ID)

		// Build prompt
		b.Grow(bufferLength)
		if config.Prefix != "" {
			b.WriteString(config.Prefix)
			b.WriteString(", ")
		}
		tagsToPrompt(&b, r, post, stripTags)
		if config.Postfix != "" {
			b.WriteString(", ")
			b.WriteString(config.Postfix)
		}
		if b.Len() > bufferLength {
			numTooLong++
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
		case <-time.After(config.PauseTime / 2.0):
		}

		slog.Info("sending generateImages", "count", config.BatchCount)
		if err := client.SendGenerate(ctx, config.BatchCount); err != nil {
			slog.Error("sendGenerateImages failed", "error", err)
			continue
		}

		if i < len(posts)-1 {
			slog.Info("waiting...")
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(config.PauseTime):
			}
		}
	}
	if numTooLong > 0 {
		slog.Warn("some prompts were too long", "count", numTooLong)
	}
	return nil
}
