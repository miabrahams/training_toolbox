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

	"github.com/knadh/koanf/v2"
	_ "github.com/marcboeker/go-duckdb"

	"training_toolbox/internal/client"
	"training_toolbox/internal/config"
	"training_toolbox/internal/database"
)

func main() {
	if err := run_main(); err != nil {
		slog.Error(err.Error())
		os.Exit(1)
	}
}

var (
	configPath = "../config/golang-generator.yml"
)

func run_main() error {
	ctx := context.Background()
	level := slog.LevelVar{}
	logger := slog.New(
		slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: &level}),
	)
	slog.SetDefault(logger)

	k, err := config.LoadConfig(configPath)
	if err != nil {
		return err
	}

	if k.String(config.LogLevelKey) != "" {
		levelReader := slog.Level(0)
		err := levelReader.UnmarshalText([]byte(k.String(config.LogLevelKey)))
		if err != nil {
			return fmt.Errorf("invalid log level: %w", err)
		}
		level.Set(levelReader)
	}

	now := time.Now()
	posts, err := runDatabaseOperations(ctx, k)
	if err != nil {
		return fmt.Errorf("load posts: %w", err)
	}
	slog.Info("loaded posts from database", "count", len(posts), "duration", time.Since(now))

	if k.Bool(config.DryRunKey) {
		slog.Info("dry run enabled, not sending posts.")
		for _, post := range posts {
			slog.Info("post data", "id", post.ID, "tags", post.Tags)
		}
		slog.Info("posts would be sent to ComfyAPIClient", "count", len(posts))
		return nil
	}

	genConfig := buildGenerationOptions(k)
	client := client.ComfyAPIClient{
		BaseURL: k.String(config.ComfyUrlKey),
	}
	client.Init()
	if err := sendPosts(ctx, client, genConfig, posts); err != nil {
		return fmt.Errorf("send posts: %w", err)
	}
	return nil
}

func loadDB(dbPath string) (*sqlx.DB, error) {
	// DuckDB creates a new database if file doesn't exist
	if _, err := os.Stat(dbPath); err != nil {
		return nil, fmt.Errorf("error locating database file %s: %w", dbPath, err)
	}

	return sqlx.Open("duckdb", dbPath+"?access_mode=read_only")
}

func buildFindPostsOptions(k *koanf.Koanf) database.FindPostsOptions {
	var minScore *int
	if k.Exists(config.MinScoreKey) {
		val := k.Int(config.MinScoreKey)
		minScore = &val
	}
	var minFavs *int
	if k.Exists(config.MinFavsKey) {
		val := k.Int(config.MinFavsKey)
		minFavs = &val
	}
	return database.FindPostsOptions{
		Tags:        k.Strings(config.SearchTagsKey),
		ExcludeTags: k.Strings(config.ExcludeTagsKey),
		MinScore:    minScore,
		MinFavs:     minFavs,
		Random:      true,
		Limit:       k.Int(config.LimitKey),
		DebugQuery:  k.Bool(config.SearchDebugKey),
	}
}

// runDatabaseOperations connects to the DB and runs standard queries. It closes the database when finished.
// TODO: When this becomes an interactive app, add dynamic database release
func runDatabaseOperations(ctx context.Context, k *koanf.Koanf) ([]database.TaggedPost, error) {
	dbPath := k.String(config.DBPathKey)
	db, err := loadDB(dbPath)
	if err != nil {
		return nil, fmt.Errorf("load db: %w", err)
	}
	defer db.Close()
	slog.Info("connected to database", "path", k.String(config.DBPathKey))

	if k.Bool(config.DBDebugKey) {
		database.DBDump(ctx, db)
	}

	options := buildFindPostsOptions(k)
	if k.Bool(config.DBDebugKey) {
		database.DBDump(ctx, db)
	}

	// Show count if enabled
	if k.Bool(config.ShowCountKey) {
		count, err := database.CountPostsWithAllTags(ctx, db, options)
		if err != nil {
			return nil, fmt.Errorf("count matching posts: %w", err)
		}
		slog.Info("total posts matching search criteria", "count", count)
	}

	taggedPosts, err := database.FindPostsWithAllTags(ctx, db, options)
	if err != nil {
		return nil, fmt.Errorf("find tag string: %w", err)
	}
	return taggedPosts, nil
}

func tagsToPrompt(b *strings.Builder, r *strings.Replacer, post database.TaggedPost, stripTags map[string]struct{}) {
	rand.Shuffle(len(post.Tags), func(i int, j int) {
		post.Tags[i], post.Tags[j] = post.Tags[j], post.Tags[i]
	})

	wroteTag := false
	for i, tag := range post.Tags {
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

type GenerationOptions struct {
	BatchCount int
	PauseTime  time.Duration
	Prefix     string
	Postfix    string
	StripTags  []string
	AddRating  bool
}

const bufferLength = 4096

func buildGenerationOptions(k *koanf.Koanf) GenerationOptions {
	genOpts := GenerationOptions{
		BatchCount: k.Int(config.GenBatchCountKey),
		PauseTime:  k.Duration(config.ComfyPauseTimeKey),
		Prefix:     "",
		Postfix:    "",
		StripTags:  k.Strings(config.GenStripTagsKey),
		AddRating:  k.Bool(config.GenAddRatingKey),
	}

	// merge style config
	style := k.String(config.StyleKey)
	if style != "" {
		styleConfigKey := fmt.Sprintf("%s.%s", config.StylesKey, style)
		if k.Exists(styleConfigKey) {
			if prefix := k.String(styleConfigKey + config.PrefixKey); prefix != "" {
				genOpts.Prefix = prefix
			}
			if postfix := k.String(styleConfigKey + config.PostfixKey); postfix != "" {
				genOpts.Postfix = postfix
			}
			stripTags := k.Strings(styleConfigKey + config.StripTagsKey)
			genOpts.StripTags = append(genOpts.StripTags, stripTags...)
		}
	}

	return genOpts
}

type PromptBuilderOptions struct {
	GenerationOptions
	Replacer  *strings.Replacer
	StripTags map[string]struct{}
}

func buildPrompt(post database.TaggedPost, config PromptBuilderOptions) string {
	b := strings.Builder{}
	b.Grow(bufferLength)
	if config.Prefix != "" {
		b.WriteString(config.Prefix)
		b.WriteString(", ")
	}

	postCopy := post
	if config.AddRating {
		postCopy.Tags = append(slices.Clone(postCopy.Tags), database.Ratings[post.Rating])
	} else {
		postCopy.Tags = slices.Clone(postCopy.Tags)
	}

	tagsToPrompt(&b, config.Replacer, post, config.StripTags)

	if config.Postfix != "" {
		b.WriteString(", ")
		b.WriteString(config.Postfix)
	}

	prompt := b.String()
	b.Reset()

	return prompt
}

func sendPosts(ctx context.Context, client client.ComfyAPIClient, config GenerationOptions, posts []database.TaggedPost) error {
	promptOpts := PromptBuilderOptions{
		GenerationOptions: config,
	}
	promptOpts.Replacer = strings.NewReplacer("_", " ", "(", "\\(", ")", "\\)")
	promptOpts.StripTags = make(map[string]struct{}, len(config.StripTags))
	for _, tag := range config.StripTags {
		promptOpts.StripTags[tag] = struct{}{}
	}

	numTooLong := 0

	for i, post := range posts {
		slog.Info("processing post", "index", i, "post", post.ID)

		prompt := buildPrompt(post, promptOpts)

		slog.Debug("sending promptReplace", "prompt", prompt, "width", post.ImageWidth, "height", post.ImageHeight)
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
