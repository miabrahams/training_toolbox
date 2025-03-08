import gradio as gr
from pathlib import Path

# Import tab components
from src.ui.tag_analysis_tab import create_tag_analysis_tab
from src.ui.frame_extractor_tab import create_frame_extractor_tab
from src.ui.prompt_search_tab import create_prompt_search_tab
from src.tag_analyzer.prompt_data import initialize_prompt_data

# Initialize prompt data first so it can be shared
db_path = Path("data/prompts.sqlite")
data_dir = Path("data")

with gr.Blocks() as app:
    # Initialize prompt data with progress bar
    with gr.Row():
        init_status = gr.Textbox(label="Initialization Status", value="Initializing...")

    # Define a simple function to initialize data with a progress bar
    def init_data(progress=gr.Progress()):
        prompt_data, db = initialize_prompt_data(db_path, progress)
        return "Data loaded successfully!"

    # Initialize data and store the result
    init_result = init_data()
    init_status.value = init_result

    # Get prompt_data and db for use in tabs
    prompt_data, db = initialize_prompt_data(db_path)

    with gr.Tabs():
        # Frame extractor tab
        frame_extractor_components = create_frame_extractor_tab()

        # Add tag analysis tab with prompt data
        tag_analysis_components = create_tag_analysis_tab(prompt_data, db, data_dir)

        # Add prompt search tab with prompt data
        prompt_search_components = create_prompt_search_tab(prompt_data, db, data_dir)

if __name__ == "__main__":
    app.launch()