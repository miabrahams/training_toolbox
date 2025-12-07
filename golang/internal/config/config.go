package config

import (
	"time"

	"github.com/knadh/koanf/parsers/yaml"
	"github.com/knadh/koanf/providers/confmap"
	"github.com/knadh/koanf/providers/file"
	"github.com/knadh/koanf/v2"
	_ "github.com/marcboeker/go-duckdb"
)

const (
	ComfyUrlKey       = "comfy.url"
	ComfyPauseTimeKey = "comfy.pause_time"
	DBPathKey         = "db.path"
	DBDebugKey        = "diagnostics.database"
	SearchDebugKey    = "diagnostics.search"
	DryRunKey         = "diagnostics.dry_run"
	GenBatchCountKey  = "generations.batch_count"
	GenAddRatingKey   = "generations.add_rating"
	StyleKey          = "generations.style"
	GenStripTagsKey   = "generations.strip_tags"
	SearchTagsKey     = "search.tags"
	ExcludeTagsKey    = "search.exclude_tags"
	MinScoreKey       = "search.min_score"
	MinFavsKey        = "search.min_favs"
	LimitKey          = "search.limit"
	RandomizeKey      = "search.randomize"
	ShowCountKey      = "search.show_count"
	LogLevelKey       = "log.level"

	ExtractPathsKey = "extract_paths"

	StylesKey    = "styles"
	PrefixKey    = ".prefix"
	PostfixKey   = ".postfix"
	StripTagsKey = ".strip_tags"
)

func LoadConfig(path string) (Config, error) {
	k := koanf.New(".")

	defaults := map[string]any{
		ComfyUrlKey:       "http://localhost:8188",
		GenBatchCountKey:  2,
		ComfyPauseTimeKey: time.Second * 5,
		DBDebugKey:        false,
	}

	k.Load(confmap.Provider(defaults, "."), nil)

	err := k.Load(file.Provider(path), yaml.Parser())
	return Config{k}, err
}

type Config struct {
	k *koanf.Koanf
}

// High-level getters to avoid referencing raw keys throughout the code.

func (c Config) LogLevel() string              { return c.k.String(LogLevelKey) }
func (c Config) DryRun() bool                  { return c.k.Bool(DryRunKey) }
func (c Config) ComfyURL() string              { return c.k.String(ComfyUrlKey) }
func (c Config) ComfyPauseTime() time.Duration { return c.k.Duration(ComfyPauseTimeKey) }

func (c Config) DBPath() string { return c.k.String(DBPathKey) }
func (c Config) DBDebug() bool  { return c.k.Bool(DBDebugKey) }

func (c Config) SearchDebug() bool     { return c.k.Bool(SearchDebugKey) }
func (c Config) ShowCount() bool       { return c.k.Bool(ShowCountKey) }
func (c Config) SearchTags() []string  { return c.k.Strings(SearchTagsKey) }
func (c Config) ExcludeTags() []string { return c.k.Strings(ExcludeTagsKey) }
func (c Config) Limit() int            { return c.k.Int(LimitKey) }
func (c Config) Randomize() bool       { return c.k.Bool(RandomizeKey) }

func (c Config) PromptExtractPaths() []string {
	return c.k.Strings(ExtractPathsKey)
}

func (c Config) MinScore() *int {
	if c.k.Exists(MinScoreKey) {
		v := c.k.Int(MinScoreKey)
		return &v
	}
	return nil
}
func (c Config) MinFavs() *int {
	if c.k.Exists(MinFavsKey) {
		v := c.k.Int(MinFavsKey)
		return &v
	}
	return nil
}

func (c Config) GenBatchCount() int     { return c.k.Int(GenBatchCountKey) }
func (c Config) GenStripTags() []string { return c.k.Strings(GenStripTagsKey) }
func (c Config) GenAddRating() bool     { return c.k.Bool(GenAddRatingKey) }
func (c Config) Style() string          { return c.k.String(StyleKey) }

// Style option helpers
func (c Config) StylePrefix(style string) string {
	return c.k.String(StylesKey + "." + style + PrefixKey)
}
func (c Config) StylePostfix(style string) string {
	return c.k.String(StylesKey + "." + style + PostfixKey)
}
func (c Config) StyleStripTags(style string) []string {
	return c.k.Strings(StylesKey + "." + style + StripTagsKey)
}
