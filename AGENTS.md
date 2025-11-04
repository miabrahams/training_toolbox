# Training Toolbox Agents

The repository hosts multiple automation-oriented tools ("agents") that support prompt analysis, dataset curation, and generation workflows. This guide catalogs each agent, how it is wired, and what it expects so you can pick the right entry point quickly.

## Quick Index

| Agent | Mode | Purpose | Primary Entry Points |
| --- | --- | --- | --- |
| Tag Analysis UI | Gradio app | Inspect prompts, clusters, metadata | `ui.py:1`, `src/ui/*.py` |
| Prompt Analyzer CLI | Command line | Batch analytics + reports | `cli.py:1` |
| Tag Analysis Core | Library | Embeddings, clustering, search | `src/tag_analyzer/tag_analyzer.py:1` |
| Caption Cleanup | Script | LLM-based caption sanitization | `captioner/process_captions.py:20` |
| Caption Collator | Script | Post-process + merge caption text | `captioner/collate_captions.py:1` |
| Caption Embeddings | Script | Generate OpenAI embedding parquet | `captioner/make_embeddings.py:1` |
| Video Frame Extractor | Gradio tab | Slice videos into frames | `src/ui/frame_extractor_tab.py:1`, `lib/ffmpeg_frames.py:6` |
| Duplicate Inspector | Python toolkit | Detect near-duplicate imagery | `src/DuplicateInspector/FindDuplicates.py:1` |
| Tag Version Control | Script | SQLite-backed caption history | `tag-versions/tag-version-control.py:10` |
| Comfy Metadata Toolkit | Library/script | Parse ComfyUI prompt/workflow data | `lib/comfy_analysis.py:1`, `comfy.py:1` |
| DB Diagnostics | Script | Inspect prompt database health | `tools/check_db.py:1` |
| Discord Prompt Analyzer | Script | Parse Discord exports for prompts | `src/training/prompt_analyzer.py:1` |
| DuckDB Prompt Loader | Go CLI | Filter/tag posts and call Comfy API | `golang/cmd/duckdb/main.go:1` |
| Unused Token Finder | Notebook | CLIP vocab coverage audit | `unused_tag_finder/unused_tokens.ipynb` |
| YouTube Transcript Agent | Notebook | Download & process transcripts | `youtube_transcript/youtube_transcript.ipynb` |
| Numerical Explorations | Notebooks | LoRA/color/latent studies | `numerical/*.ipynb` |

## Tag Analysis Suite

### Gradio UI
- Orchestrated in `ui.py:1`, which wires tabs for prompt extraction, frame extraction, clustering, and direct SQL views.
- Tabs are modular (`src/ui/comfy_prompt_extractor_tab.py:1`, `frame_extractor_tab.py:1`, `tag_analysis_tab.py:1`, `prompt_search_tab.py:1`, `direct_search_tab.py:1`) so new functionality drops in by adding another tab factory.
- Initialization bridges the SQLite prompt DB and cached analysis data (`ui.py:75-121`). Users set the database path, click **Load Data**, and the app lazily computes embeddings unless `Compute Analysis` is triggered from the Tag Analysis tab.
- Dependencies: Gradio, SentenceTransformers, UMAP, HDBSCAN, FFmpeg, numpy. Install with `uv sync` (add `--extra duplicates` when the duplicate inspector features are needed).

### Prompt Analyzer CLI
- Located in `cli.py:1`; exposes `summary`, `visualize`, `analyze-dir`, `tags`, and `modifiers` subcommands via `argparse` (`cli.py:8-63`).
- Shares initialization with the UI (`cli.py:71-81`), then dispatches to rich reporting helpers (`cli.py:128-200`).
- Ideal when you need reproducible, scriptable analytics or to export plots via Matplotlib.

### Tag Analysis Core Library
- `TagAnalyzer` (`src/tag_analyzer/tag_analyzer.py:32-215`) encapsulates embedding generation (SentenceTransformer `all-mpnet-base-v2`, `src/tag_analyzer/tag_analyzer.py:150-155`), dimensionality reduction (UMAP, `src/tag_analyzer/tag_analyzer.py:156-164`), and clustering (HDBSCAN, `src/tag_analyzer/tag_analyzer.py:166-173`).
- Provides high-level calls for searching prompts, summarizing clusters, visualizing embeddings, diffing prompts, and inspecting directory-specific contributions.
- Analysis artifacts (embeddings, reduced embeddings, clusters) persist through `TagAnalysisData` so repeated runs reuse cached results.

### Database Utilities
- `TagDatabase` handlers live in `src/tag_analyzer/database.py` (for CRUD and state tracking) and pair with `tools/check_db.py:11-87`, which prints schema info and sample prompts.
- Use `python tools/check_db.py data/prompts.sqlite` to confirm metadata extraction before running heavier analyses.

## Caption Processing Agents

### Caption Cleanup (LLM Moderation)
- Implemented in `captioner/process_captions.py:20-130`. A `CaptionProcessor` client streams captions through Google Gemini (`captioner/process_captions.py:32-70`) and routes flagged items to `./data/errors` while clean ones land in `./data/output`.
- Requires `CAPTION_API_KEY` in the environment (`captioner/process_captions.py:121-128`) and the `captioner` extra (`uv sync --extra captioner`).
- Tweak the input/output directories at the top of the script, then run `uv run --extra captioner python captioner/process_captions.py`.

### Caption Collator
- `captioner/collate_captions.py:1-74` walks processed captions, applies regex-based cleanups, and produces a unified `data/collated_captions.txt` file for downstream embedding.
- Set `POSTPROCESS = True` to enable normalization rules that collapse repetitive descriptors and remove redundant phrases.

### Caption Embedding Builder
- `captioner/make_embeddings.py:1-132` loads `config.yml`/`secrets.yml` via `lib/config.py:5-11`, then batches the collated captions through the OpenAI embeddings API (`captioner/make_embeddings.py:70-113`).
- Outputs a Parquet file containing text + embedding vectors defined in `config.yml` under `embeddings.parquet`.
- Requires either `OPENAI_API_KEY` in environment or in `secrets.yml`.

### Caption Version Control
- `tag-versions/tag-version-control.py:10-200` tracks caption revisions inside SQLite, deduplicating updates by SHA256 hash and enabling bulk restore operations.
- Usage pattern: collect target caption paths, call `CaptionVersionControl.bulk_add_versions` for journaling, and later `bulk_restore_versions` or `bulk_get_versions` for auditing.

## Vision & Metadata Agents

### Video Frame Extractor
- UI tab defined in `src/ui/frame_extractor_tab.py:1-136` uses FFmpeg to sample frames from selected videos, handling WSL path conversion via `lib/wsl_utils.py:5-68`.
- Backend extraction pipeline lives in `lib/ffmpeg_frames.py:6-37`; CLI usage supported through the same module (`lib/ffmpeg_frames.py:40-64`).

### Duplicate Inspector
- Toolkit under `src/DuplicateInspector/` provides multiple strategies (hash-based, CuPy-accelerated) for finding duplicate images across datasets. Enable the extras with `uv sync --extra duplicates`.
- Scripts like `FindDuplicates.py` expose command-line entry points; `InspectDuplicates.ipynb` offers exploratory analysis.

### Comfy Metadata Toolkit
- `lib/comfy_analysis.py:1-188` defines schema readers that parse ComfyUI prompt/workflow metadata, extracting prompt text, LoRA usage, and inference settings.
- `comfy.py:1-27` demonstrates iterating through `data/` images and building `ComfyImage` objects for ad-hoc inspection.
- These utilities power both the UI prompt extractor tab and database checks.

### Discord Prompt Analyzer
- `src/training/prompt_analyzer.py:1-198` downloads Discord-exported images, extracts embedded metadata (`src/training/prompt_analyzer.py:94-118`), and builds a Pandas dataset with prompt statistics and samples.
- Useful for auditing community-sourced prompts before import into the main analyzer.

### DuckDB Prompt Loader (Go)
- Go pipeline at `golang/cmd/duckdb/main.go:1-200` reads configuration (`config.yml`), filters posts in DuckDB, optionally prints diagnostics, and sends batches to a Comfy API client.
- Shared helpers live under `golang/internal/` (SQLC database bindings, prompt parsing) with builds driven by `golang/Makefile` and `Makefile:3-11` targets.
- Compile or run with `cd golang && go run cmd/duckdb/main.go` (requires Go 1.22+, DuckDB driver, and matching config paths).

### Database Loader CLI
- `Makefile:3-11` also exposes a `load_prompts` recipe that calls `golang/cmd/load/main.go` to ingest Comfy output directories into SQLite.

### Supporting Notebooks & Agents
- **Unused Token Finder** (`unused_tag_finder/unused_tokens.ipynb`) cross-references `clip_vocab.json` with dataset tag usage to flag dead vocabulary.
- **YouTube Transcript Agent** (`youtube_transcript/youtube_transcript.ipynb`) pulls transcripts using the API enabled by `uv sync --extra youtube`.
- **Numerical Studies** (`numerical/*.ipynb`) contain exploratory analysis for LoRA weights, latent space inspection, and color statistics.

## Shared Configuration & Data

- `config.yml:1-115` and `secrets.yml` store shared settings: database paths, embedding outputs, generation styles, Comfy endpoints, and API keys. Python agents read them via `lib/config.py:5-11`; Go agents use the same file through Koanf.
- Datasets and cache artifacts live in `data/` (SQLite DB, numpy arrays, scraped prompts) and `models/` (e.g., EVA CLIP weights). Keep these directories available when running analyzers.
- Global Python dependencies now live in `pyproject.toml`; run `./install-requirements.sh` (a thin wrapper over `uv sync`) to create or update the environment. Pass extras such as `--extra captioner duplicates` when specialized agents are required.

## Getting Started

1. **Install dependencies** with UV (see `install-requirements.sh:1-5`) or adapt to your package manager.
2. **Prepare secrets** in `secrets.yml` or environment variables (`CAPTION_API_KEY`, `OPENAI_API_KEY`, Comfy credentials).
3. **Validate the prompt database** using `python tools/check_db.py data/prompts.sqlite`.
4. **Launch the Gradio UI** via `uv run training-toolbox-ui` (or `python ui.py`) or run targeted agents (CLI/Go) as needed.

Each agent is designed to be composable: start with database validation, move through caption cleanup and embedding, inspect clusters visually, and finally automate generation through the Go pipeline.
