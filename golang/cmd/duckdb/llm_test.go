package main

import (
	"os"
	"testing"

	"github.com/openai/openai-go/v2"
	"github.com/stretchr/testify/require"
)

func TestQuery(t *testing.T) {
	t.Skip()
	t.Log(os.Getwd())
	secrets, err := LoadSecrets("./../../../secrets.yml")
	require.NoError(t, err, "Failed to load secrets")

	apiKey := secrets.String("DEEPSEEK_API_KEY")
	require.NotEmpty(t, apiKey, "DEEPSEEK_API_KEY is required")

	client, err := setup_client(apiKey)
	require.NoError(t, err, "Failed to setup client")
	chatCompletion, err := client.Chat.Completions.New(t.Context(), openai.ChatCompletionNewParams{
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage("Say this is a test"),
		},
		Model: "deepseek-chat",
	})
	require.NoError(t, err, "Failed to create chat completion")
	t.Log("Query executed successfully:", chatCompletion.Choices[0].Message.Content)
}
