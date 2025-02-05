package png

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"
	"strconv"
	"strings"
)

func ExtractTextChunks(filename string) (map[string]string, error) {
	// Read the file
	data, err := os.ReadFile(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to read file: %w", err)
	}

	// Check PNG signature: 137 80 78 71 13 10 26 10
	if len(data) < 8 ||
		data[0] != 137 || data[1] != 80 || data[2] != 78 || data[3] != 71 ||
		data[4] != 13 || data[5] != 10 || data[6] != 26 || data[7] != 10 {
		return nil, fmt.Errorf("file is not a valid PNG")
	}

	// Iterate over chunks starting after signature
	offset := 8
	result := make(map[string]string)
	for offset+8 <= len(data) {
		// Read length (4 bytes, big endian)
		length := binary.BigEndian.Uint32(data[offset : offset+4])
		if offset+8+int(length)+4 > len(data) {
			break // not enough data for the chunk
		}
		// Read chunk type (next 4 bytes)
		chunkType := string(data[offset+4 : offset+8])
		// chunk data starts after 8 header bytes
		chunkDataStart := offset + 8
		chunkDataEnd := chunkDataStart + int(length)
		if chunkType == "tEXt" {
			nullIndex := bytes.IndexByte(data[chunkDataStart:chunkDataEnd], 0)
			if nullIndex != -1 {
				keyword := string(data[chunkDataStart : chunkDataStart+nullIndex])
				prompt := string(data[chunkDataStart+nullIndex+1 : chunkDataEnd])
				result[keyword] = prompt
			}
		} else if chunkType == "IDAT" {
			break
		}
		// Move to the next chunk: chunk header (8 bytes) + data + CRC (4 bytes)
		offset += int(length) + 12
	}

	if len(result) > 0 {
		return result, nil
	}
	return nil, fmt.Errorf("no tEXt chunks found")
}

// UnescapeJSON removes escape sequences from a JSON string extracted from a tEXt chunk.
// It wraps the string in quotes and uses strconv.Unquote.
func UnescapeJSON(data string) (string, error) {
	// By wrapping data in quotes, Unquote will convert escaped sequences.
	unescaped, err := strconv.Unquote(data)
	if err != nil {
		return "", fmt.Errorf("failed to unescape JSON: %w", err)
	}
	return unescaped, nil
}

func UnescapeStrJank(data string) string {
	// Just replace `\"` with `"` and `\\` with `\`
	return strings.ReplaceAll(data, "\n", "")
}

// ParseChunkJSON unescapes the given chunk content and parses the result as JSON.
// It returns the JSON data as a map.
func ParseChunkJSON(data string) (map[string]any, error) {
	var result map[string]interface{}
	if err := json.Unmarshal([]byte(data), &result); err != nil {
		return nil, fmt.Errorf("failed to parse JSON: %w", err)
	}
	return result, nil
}
