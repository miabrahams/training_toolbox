package parser

import (
	"fmt"
)

type Prompt map[string]any

func ParseChunk(chunk string) (Prompt, error) {
	// Parse the JSON content of the prompt.
	parsed, err := ParseChunkJSON(chunk)
	if err != nil {
		return nil, fmt.Errorf("failed to parse chunk JSON: %w", err)
	}

	return parsed, nil
}

func ExtractFileChunks(filepath string) (string, string, error) {
	// Extract the tEXt chunks.
	chunks, err := ExtractTextChunks(filepath)
	if err != nil {
		return "", "", fmt.Errorf("error extracting chunks: %w", err)
	}

	promptRaw, ok := chunks["prompt"]
	if !ok {
		return "", "", fmt.Errorf("no 'prompt' chunk found")
	}

	workflowRaw, ok := chunks["workflow"]
	if !ok {
		return "", "", fmt.Errorf("no 'workflow' chunk found")
	}

	return promptRaw, workflowRaw, err
}
