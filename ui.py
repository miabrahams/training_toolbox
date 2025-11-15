import gradio as gr
import numpy as np
from pathlib import Path

# Import tab components
from src.ui.tag_analysis_tab import create_tag_analysis_tab
from src.ui.frame_extractor_tab import create_frame_extractor_tab
from src.ui.prompt_search_tab import create_prompt_search_tab
from src.ui.direct_search_tab import create_direct_search_tab
from src.ui.comfy_prompt_extractor_tab import create_comfy_prompt_extractor_tab
from src.controllers.prompts.prompt_data import PromptData
from src.controllers.prompts.processor import PromptProcessor
from src.controllers.tags.tag_cluster_analyzer import create_analyzer
from src.db.prompt_database import PromptDatabase

from src.lib.config import load_settings


# The analyzer is designed to support GUI and TUI front-ends, so initialization functions are separated.
def initialize_analyzer(data_dir: Path, prompt_data: PromptData, db: PromptDatabase,
                        force_recompute=False, progress=gr.Progress()):
    """Initialize the analyzer with given paths and display progress"""
    try:
        analyzer = create_analyzer(
            data_dir=data_dir,
            prompt_data=prompt_data,
            compute_analysis=force_recompute,
            progress=progress
        )

        if analyzer is not None:
            if analyzer.analysis is not None:
                n_clusters = len(np.unique(analyzer.clusters)) - (1 if -1 in analyzer.clusters else 0)
                return f"Loaded {len(analyzer.prompt_texts)} prompts with {n_clusters} clusters.", analyzer
            else:
                return "Analyzer created but no clusters computed yet. Use 'Compute Analysis' button in Tag Analysis tab.", analyzer
        else:
            return "Error: Failed to initialize the tag analyzer. Check the database path.", None
    except Exception as e:
        return f"Error: {str(e)}", None


settings = load_settings()
ui_defaults = settings.get("ui.defaults", {})

with gr.Blocks() as app:
    # Top level configuration section
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("## Training Toolbox")
            init_status = gr.Textbox(label="Initialization Status", value="Not initialized")

        with gr.Column(scale=3):
            with gr.Row():
                db_path = gr.Textbox(
                    label="Database Path",
                    value=ui_defaults.get("db_path"),
                    info="Path to SQLite database with prompts"
                )

                data_dir = gr.Textbox(
                    label="Data Directory",
                    value=ui_defaults.get("data_dir"),
                    info="Directory to save/load analysis data"
                )

            load_btn = gr.Button("Load Data", variant="primary")

    # Store analyzer state for all components to access
    analyzer_state = gr.State(None)
    prompt_data_state = gr.State(None)
    db_state = gr.State(None)

    # Connect load button to init function
    def handle_load(db_path_str, data_dir_str, progress=gr.Progress()):
        try:
            # Convert strings to Path objects
            db_path = Path(db_path_str)
            data_path = Path(data_dir_str)

            # Initialize prompt data with progress updates
            progress(0.1, "Loading prompt data...")
            # Initialize via controller (keeps SQL out of PromptData)
            db = PromptDatabase(db_path)
            processor = PromptProcessor(db)
            processor.process_new_prompts(progress)
            prompt_data = processor.load_prompts()
            progress(0.5, "Creating analyzer...")

            # Initialize analyzer without computing analysis yet
            status_msg, analyzer = initialize_analyzer(
                data_dir=data_path,
                prompt_data=prompt_data,
                db=db,
                progress=progress
            )

            progress(1.0, "Initialization complete!")

            return status_msg, prompt_data, db, analyzer
        except Exception as e:
            return f"Error: {str(e)}", None, None, None

    load_btn.click(
        fn=handle_load,
        inputs=[db_path, data_dir],
        outputs=[init_status, prompt_data_state, db_state, analyzer_state]
    )

    with gr.Tabs():
        # ComfyUI Prompt Extractor tab (no dependencies needed)
        comfy_extractor_components = create_comfy_prompt_extractor_tab()

        # Frame extractor tab
        frame_extractor_components = create_frame_extractor_tab()

        # Add tag analysis tab with analyzer_state (will be updated when data is loaded)
        tag_analysis_components = create_tag_analysis_tab(analyzer_state)

        # Add prompt search tab with analyzer_state
        prompt_search_components = create_prompt_search_tab(analyzer_state)

        # Add direct search tab
        direct_search_components = create_direct_search_tab(db_state)

def main(server_port: int | None = None) -> None:
    """Launch the Gradio interface for the training toolbox."""
    port = server_port or settings.get("ui.server.port", 7000)
    app.launch(server_port=port)


if __name__ == "__main__":
    main()
