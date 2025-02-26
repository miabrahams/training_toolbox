import gradio as gr
import os
import subprocess
from pathlib import Path
from ffmpeg_frames import extract_frames

def wsl_path_to_windows(path):
    """Convert a WSL path to Windows path"""
    if path.startswith('/mnt/'):
        # drive = path[5:6].upper()
        return subprocess.check_output(['wslpath', '-w', path]).decode().strip()
    return path

def windows_path_to_wsl(path):
    """Convert a Windows path to WSL path"""
    if ':' in path:  # Windows path detected
        return subprocess.check_output(['wslpath', path]).decode().strip()
    return path

def is_wsl():
    """Check if running under WSL"""
    return 'microsoft' in open('/proc/version').read().lower() if os.path.exists('/proc/version') else False

def list_directories(base_path="/mnt/c/Users"):
    """Return a list of directories at the given path"""
    try:
        directories = [d for d in Path(base_path).iterdir() if d.is_dir()]
        return sorted([str(d) for d in directories])
    except Exception as e:
        return [f"Error: {str(e)}"]

def list_videos(directory):
    """List video files in the given directory"""
    try:
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
        video_files = []
        for ext in video_extensions:
            video_files.extend([str(f) for f in Path(directory).glob(f'*{ext}')])
        return sorted(video_files)
    except Exception as e:
        return [f"Error: {str(e)}"]

def process_videos(selected_dir, selected_videos, output_dir, num_frames=5):
    """Process selected videos from the directory"""
    if not selected_dir or not selected_videos:
        return "Please select a directory and at least one video"

    if not output_dir:
        output_dir = os.path.join(selected_dir, "extracted_frames")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    results = []
    for video_path in selected_videos:
        try:
            frames_dir = extract_frames(
                Path(video_path),
                output_dir,
                max_frames=num_frames
            )
            results.append(f"Extracted {num_frames} frames from {video_path} to {frames_dir}")
        except Exception as e:
            results.append(f"Error processing {video_path}: {str(e)}")

    return "\n".join(results)

def navigate_to_parent(current_path):
    """Navigate to parent directory"""
    parent = str(Path(current_path).parent)
    return parent, list_directories(parent)

with gr.Blocks() as app:
    gr.Markdown("# Video Frame Extractor")

    current_path = gr.State("/mnt/c/Users")

    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("## Navigation")
            current_dir = gr.Textbox(value="/mnt/c/Users", label="Current Directory")
            up_button = gr.Button("Go Up")
            dir_dropdown = gr.Dropdown(choices=list_directories("/mnt/c/Users"), label="Select Subdirectory")

            with gr.Row():
                refresh_btn = gr.Button("Refresh")

            video_selector = gr.CheckboxGroup(choices=[], label="Select Videos")

            output_dir = gr.Textbox(label="Output Directory (leave empty for default)")
            num_frames = gr.Slider(minimum=1, maximum=30, value=5, step=1, label="Number of frames to extract")

            process_btn = gr.Button("Extract Frames", variant="primary")

        with gr.Column(scale=3):
            output = gr.Textbox(label="Status", lines=10)

    # Update directory listing when going up
    up_button.click(
        navigate_to_parent,
        [current_dir],
        [current_dir, dir_dropdown]
    )

    # Update current directory when selecting a subdirectory
    def update_current_dir(selected_dir, current):
        if not selected_dir:
            return current, []
        new_path = os.path.join(current, os.path.basename(selected_dir))
        return new_path, list_videos(new_path)

    dir_dropdown.change(
        update_current_dir,
        [dir_dropdown, current_dir],
        [current_dir, video_selector]
    )

    # Refresh current directory
    refresh_btn.click(
        lambda x: (x, list_videos(x)),
        [current_dir],
        [current_dir, video_selector]
    )

    # Process videos button
    process_btn.click(
        process_videos,
        [current_dir, video_selector, output_dir, num_frames],
        output
    )

if __name__ == "__main__":
    app.launch()
