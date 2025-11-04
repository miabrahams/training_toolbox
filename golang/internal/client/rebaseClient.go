package client

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

// All routes match the following format with differing payloads
type ComfyRebaseRequest[D any] struct {
	Event string `json:"event"`
	Data  D      `json:"data"`
}

type RebasePromptPayload struct {
	PositivePrompt string `json:"positive_prompt"`
	Resolution     struct {
		Width  int `json:"width"`
		Height int `json:"height"`
	} `json:"resolution"`
}

type RebasePromptRequest = ComfyRebaseRequest[RebasePromptPayload]

type GenerateImagePayload struct {
	Count int `json:"count"`
}
type RebaseGenerateRequest = ComfyRebaseRequest[GenerateImagePayload]

type ComfyAPIClient struct {
	BaseURL    string
	HTTPClient *http.Client
}

func (c *ComfyAPIClient) Init() {
	if c.HTTPClient == nil {
		c.HTTPClient = &http.Client{}
	}
}

func (c *ComfyAPIClient) JSONRequest(ctx context.Context, method, route string, body interface{}) (*http.Response, error) {
	var buf bytes.Buffer
	if err := json.NewEncoder(&buf).Encode(body); err != nil {
		return nil, fmt.Errorf("encode json: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, method, c.BaseURL+route, &buf)
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	res, err := c.HTTPClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("send request: %w", err)
	}
	return res, nil
}

func (c *ComfyAPIClient) SendPromptReplace(ctx context.Context, prompt string, width, height int) error {
	payload := RebasePromptRequest{
		Event: "promptReplace",
	}
	payload.Data.PositivePrompt = prompt
	payload.Data.Resolution.Width = width
	payload.Data.Resolution.Height = height

	res, err := c.JSONRequest(ctx, http.MethodPost, "/rebase/forward", payload)
	if err != nil {
		return err
	}
	defer res.Body.Close()

	if res.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(res.Body)
		return fmt.Errorf("unexpected status: %s, body: %s", res.Status, body)
	}
	return nil
}

func (c *ComfyAPIClient) SendGenerate(ctx context.Context, count int) error {
	payload := RebaseGenerateRequest{
		Event: "generateImages",
	}
	payload.Data.Count = count

	res, err := c.JSONRequest(ctx, http.MethodPost, "/rebase/forward", payload)
	if err != nil {
		return err
	}
	if res.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(res.Body)
		return fmt.Errorf("unexpected status: %s, body: %s", res.Status, body)
	}
	return nil
}
