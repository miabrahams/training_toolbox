Training toolbox for managing prompt, caption, and metadata workflows across analysis, curation, and generation.

# Toolbox UI
Gradio application orchestrated in `ui.py` with modular tabs under `src/ui/`. It wires together prompt extraction, tag analytics, frame extraction, and direct SQL browsing against the SQLite prompt database. Launch with `python ui.py`, pick a database path, then press **Load Data** to hydrate cached embeddings or trigger **Compute Analysis** for fresh UMAP/HDBSCAN clustering. Requires the dependencies in `src/requirements.txt` (installable with `uv pip install -r src/requirements.txt`).

# Unused token finder
Notebook-based audit located at `unused_tag_finder/unused_tokens.ipynb`. It cross-references dataset tags with the CLIP vocabulary to surface tokens that never appear in your training metadata. Open the notebook, point it at your tag exports, and run the provided cells to review the resulting coverage tables and plots.

# Tag version control
Script entry point `tag-versions/tag-version-control.py` maintains a SQLite-backed history for caption or prompt `.txt` files. It hashes every revision, deduplicates identical content, and allows bulk restore/export operations so you can roll back to previous captions on demand. Integrate by collecting target paths, then calling `CaptionVersionControl.bulk_add_versions` followed by `bulk_restore_versions` or `bulk_get_versions` as needed.

# Captioner
Caption processing toolkit under `captioner/` that cleans, collates, and embeds caption text. Run `captioner/process_captions.py` to stream captions through Google Gemini (requires `CAPTION_API_KEY`), optionally clean the outputs with `captioner/collate_captions.py`, and build OpenAI embedding parquet files via `captioner/make_embeddings.py` (expects `OPENAI_API_KEY`). Install requirements from `captioner/requirements.txt` before launching the pipeline.

# Youtube_transcript
Notebook `youtube_transcript/youtube_transcript.ipynb` plus its `requirements.txt` download and normalize YouTube transcripts for downstream prompt analysis. After installing dependencies, run the notebook to authenticate, pull transcripts, and export structured text suitable for caption ingestion.
