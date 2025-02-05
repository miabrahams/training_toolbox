-- name: ListPrompts :many
SELECT file_path, prompt, workflow FROM prompts;

-- name: GetTestPrompt :one
SELECT file_path, prompt, workflow FROM prompts
limit 1;

-- name: GetPromptByPath :one
SELECT file_path, prompt, workflow FROM prompts
WHERE file_path = ?;