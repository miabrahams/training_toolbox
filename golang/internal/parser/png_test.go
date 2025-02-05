package parser_test

import (
	"path/filepath"
	"testing"
	"training_toolbox/internal/parser"

	"github.com/stretchr/testify/require"
)

func TestExtractChunk(t *testing.T) {
	// Path to the test PNG file in the testdata folder.
	testFile := filepath.Join("data", "sample.png")

	// Call the function to extract the chunk.
	chunks, err := parser.ExtractTextChunks(testFile)
	require.NoError(t, err)
	require.Len(t, chunks, 2)

	// Check the extracted chunks.
	workflow_raw := chunks["workflow"]
	require.NotEmpty(t, workflow_raw)

	prompt_raw := chunks["prompt"]
	require.NotEmpty(t, prompt_raw)

	prompt, err := parser.ParseChunkJSON(prompt_raw)
	require.NoError(t, err)
	require.NotEmpty(t, prompt)
}
