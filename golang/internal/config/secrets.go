package config

import (
	"github.com/knadh/koanf/parsers/yaml"
	"github.com/knadh/koanf/providers/confmap"
	"github.com/knadh/koanf/providers/file"
	"github.com/knadh/koanf/v2"
)

const (
	DeepseekConfKey = "DEEPSEEK_API_KEY"
)

func LoadSecrets(path string) (Secrets, error) {
	k := koanf.New(".")

	defaults := map[string]any{
		DeepseekConfKey: "",
	}

	k.Load(confmap.Provider(defaults, "."), nil)

	err := k.Load(file.Provider(path), yaml.Parser())
	if err != nil {
		return Secrets{}, err
	}

	return Secrets{k}, nil
}

type Secrets struct {
	k *koanf.Koanf
}

func (s Secrets) DeepseekAPIKey() string {
	return s.k.String(DeepseekConfKey)
}
