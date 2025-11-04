Training toolbox for managing prompt, caption, and metadata workflows across analysis, curation, and generation.

# Configuration
Defaults for every agent live in `config/config.yml` (with secrets in `config/secrets.yml`).
`lib.config.get_settings()` returns a shared Dynaconf settings object that merges both files, so scripts, CLIs, and notebooks can all read consistent values (e.g. `settings.get("captioner.output_dir")`).
Override the config directory with `TRAINING_TOOLBOX_CONFIG_DIR` or add `config.local.yml`/`config.d/*.yml` for per-machine tweaks.

# Toolbox UI
Gradio application orchestrated in `ui.py` with modular tabs under `src/ui/`. It wires together prompt extraction, tag analytics, frame extraction, and direct SQL browsing against the SQLite prompt database. Sync dependencies with `uv sync` (optionally `--extra duplicates captioner` as needed), then launch with `uv run training-toolbox-ui` to load a database, hydrate cached embeddings, or trigger **Compute Analysis** for fresh UMAP/HDBSCAN clustering.

# Unused token finder
Notebook-based audit located at `unused_tag_finder/unused_tokens.ipynb`. It cross-references dataset tags with the CLIP vocabulary to surface tokens that never appear in your training metadata. Open the notebook, point it at your tag exports, and run the provided cells to review the resulting coverage tables and plots.

# Tag version control
Script entry point `tag-versions/tag-version-control.py` maintains a SQLite-backed history for caption or prompt `.txt` files. It hashes every revision, deduplicates identical content, and allows bulk restore/export operations so you can roll back to previous captions on demand. Integrate by collecting target paths, then calling `CaptionVersionControl.bulk_add_versions` followed by `bulk_restore_versions` or `bulk_get_versions` as needed.

# Captioner
Caption processing toolkit under `captioner/` that cleans, collates, and embeds caption text. Paths for the raw/processed/error directories, collated output, and embedding targets come from the `captioner.*` section in `config/config.yml`. Run `uv run --extra captioner python captioner/process_captions.py` to stream captions through Google Gemini (requires `CAPTION_API_KEY` or `captioner.api_key` in secrets), optionally clean the outputs with `captioner/collate_captions.py`, and build OpenAI embedding parquet files via `captioner/make_embeddings.py` (expects `OPENAI_API_KEY` or `openai.api_key` in secrets).

# Youtube_transcript
Notebook `youtube_transcript/youtube_transcript.ipynb` downloads and normalizes YouTube transcripts for downstream prompt analysis. Enable the notebook environment with `uv sync --extra youtube`, run the notebook to authenticate, pull transcripts, and export structured text suitable for caption ingestion.
