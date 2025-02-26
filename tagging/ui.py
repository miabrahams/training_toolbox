import gradio as gr
import os
from pathlib import Path


def list_directories(base_path="/mnt/c/Users"):
    """Return a list of directories at the given path"""
    directories = [d for d in Path(base_path).iterdir() if d.is_dir()]
    return sorted([str(d) for d in directories])

def process_videos(selected_dir, num_frames=5):
    # Your video processing code here
    return f"Processing videos in {selected_dir} with {num_frames} frames each"

with gr.Blocks() as app:
    gr.Markdown("# Video Frame Extractor")

    with gr.Row():
        with gr.Column():
            initial_dirs = list_directories()
            dir_dropdown = gr.Dropdown(choices=initial_dirs, label="Select directory")

            def update_dirs(dir_path):
                return gr.Dropdown(choices=list_directories(dir_path))

            dir_dropdown.change(update_dirs, dir_dropdown, dir_dropdown)

            num_frames = gr.Slider(minimum=1, maximum=20, value=5, step=1, label="Number of frames")
            process_btn = gr.Button("Extract Frames")

        with gr.Column():
            output = gr.Textbox(label="Status")

    process_btn.click(process_videos, [dir_dropdown, num_frames], output)

app.launch()
