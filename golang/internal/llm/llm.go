package main

import (
	"context"

	"github.com/openai/openai-go/v2"
	"github.com/openai/openai-go/v2/option"
)

func setup_client(apiKey string) (openai.Client, error) {
	client := openai.NewClient(
		option.WithAPIKey(apiKey),
		option.WithBaseURL("https://api.deepseek.com/v1"),
	)
	return client, nil
}

func RunQuery(apiKey string) error {
	client, err := setup_client(apiKey)
	if err != nil {
		return err
	}
	chatCompletion, err := client.Chat.Completions.New(context.TODO(), openai.ChatCompletionNewParams{
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage("Say this is a test"),
		},
		Model: "deepseek-chat",
	})
	if err != nil {
		return err
	}
	println(chatCompletion.Choices[0].Message.Content)
	return nil
}
