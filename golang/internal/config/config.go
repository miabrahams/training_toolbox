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
	StylesKey         = "styles"
	PrefixKey         = ".prefix"
	PostfixKey        = ".postfix"
	StripTagsKey      = ".strip_tags"
)

func LoadConfig(path string) (*koanf.Koanf, error) {
	k := koanf.New(".")

	defaults := map[string]any{
		ComfyUrlKey:       "http://localhost:8188",
		GenBatchCountKey:  2,
		ComfyPauseTimeKey: time.Second * 5,
		DBDebugKey:        false,
	}

	k.Load(confmap.Provider(defaults, "."), nil)

	err := k.Load(file.Provider(path), yaml.Parser())
	return k, err
}

func LoadSecrets(path string) (*koanf.Koanf, error) {
	k := koanf.New(".")

	defaults := map[string]any{
		"DEEPSEEK_API_KEY": "",
	}

	k.Load(confmap.Provider(defaults, "."), nil)

	err := k.Load(file.Provider(path), yaml.Parser())
	if err != nil {
		return nil, err
	}

	return k, nil
}
