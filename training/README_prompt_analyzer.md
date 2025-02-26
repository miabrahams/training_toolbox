# Discord Prompt Analyzer

A tool for analyzing AI art prompts from Discord servers. This tool works with data exported from Discord using DiscordChatExporter.

## Features

- Extract and analyze image metadata from Discord messages
- Parse Stable Diffusion, NovelAI, and other AI generation prompts
- Analyze user reactions to determine popular prompt patterns

## Setup

1. Export Discord data using DiscordChatExporter
2. Place the exported JSON files in the `discord_logs` directory
3. Run the analyzer: `python prompt_analyzer.py`

## Usage

```bash
# Basic usage
python prompt_analyzer.py

# Specify custom data path
python prompt_analyzer.py --path /path/to/data/directory

# Analyze a specific log file
python prompt_analyzer.py --file exported_log.json
```

## Requirements

- Python 3.10+
- Pandas
- Pillow
- tqdm
- requests
