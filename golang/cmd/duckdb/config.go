package main

import (
	"time"

	"github.com/knadh/koanf/parsers/yaml"
	"github.com/knadh/koanf/providers/confmap"
	"github.com/knadh/koanf/providers/file"
	"github.com/knadh/koanf/v2"
	_ "github.com/marcboeker/go-duckdb"
)

const (
	comfyUrlKey       = "comfy.url"
	dbPathKey         = "db.path"
	dbDebugKey        = "diagnostics.database"
	searchDebugKey    = "diagnostics.search"
	genBatchCountKey  = "generations.batch_count"
	genPauseTimeKey   = "generations.pause_time"
	genAddRatingKey   = "generations.add_rating"
	styleKey          = "generations.style"
	genStripTagsKey   = "generations.strip_tags"
	defaultPrefixKey  = "generations.default_prefix"
	defaultPostfixKey = "generations.default_postfix"
	searchTagsKey     = "search.tags"
	excludeTagsKey    = "search.exclude_tags"
	minScoreKey       = "search.min_score"
	minFavsKey        = "search.min_favs"
	limitKey          = "search.limit"
	randomizeKey      = "search.randomize"
	showCountKey      = "search.show_count"
	logLevelKey       = "log.level"
	styleConfigsKey   = ".styleConfigs"
	prefixKey         = ".prefix"
	postfixKey        = ".postfix"
	stripTagsKey      = ".strip_tags"
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
