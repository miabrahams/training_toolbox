package parser

import (
	"fmt"
	"training_toolbox/internal/png"
)

type Prompt map[string]any

func ParseChunk(chunk string) (Prompt, error) {
	// Parse the JSON content of the prompt.
	parsed, err := png.ParseChunkJSON(chunk)
	if err != nil {
		return nil, fmt.Errorf("failed to parse chunk JSON: %w", err)
	}

	return parsed, nil
}

/*

	prompt, err := ParseChunk(promptRaw)
	if err != nil {
		return "", "", fmt.Errorf("error parsing 'prompt' chunk: %w", err)
	}

	workflow, err := ParseChunk(workflowRaw)
	if err != nil {
		return "", "", fmt.Errorf("error parsing 'prompt' chunk: %w", err)
	}
*/

func ParseFile(filepath string) (string, string, error) {
	// Extract the tEXt chunks.
	chunks, err := png.ExtractTextChunks(filepath)
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
