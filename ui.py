import gradio as gr

# Import tab components
from src.ui import create_tag_analysis_tab
from src.ui.frame_extractor_tab import create_frame_extractor_tab

with gr.Blocks() as app:
    with gr.Tabs():
        # Frame extractor tab
        frame_extractor_components = create_frame_extractor_tab()

        # Add tag analysis tab
        tag_analysis_components = create_tag_analysis_tab()

if __name__ == "__main__":
    app.launch()
