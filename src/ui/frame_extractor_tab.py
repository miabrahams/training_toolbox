import gradio as gr
import os
from pathlib import Path
import asyncio
from src.lib.ffmpeg_frames import extract_frames
from src.lib.wsl_utils import convert_path_if_needed, is_wsl

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

# TODO: put this in a library
async def process_videos(selected_dir, selected_videos, output_dir, num_frames=5, status_callback=None):
    """Process selected videos from the directory with async updates"""
    if not selected_dir or not selected_videos:
        return "Please select a directory and at least one video"

    if not output_dir:
        output_dir = os.path.join(selected_dir, "extracted_frames")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    results = []
    total_videos = len(selected_videos)

    for i, video_path in enumerate(selected_videos, 1):
        status_msg = f"Processing video {i}/{total_videos}: {os.path.basename(video_path)}"
        if status_callback:
            await status_callback(status_msg)

        try:
            frames_dir = extract_frames(
                Path(video_path),
                output_dir,
                max_frames=num_frames
            )
            result_msg = f"✅ Extracted {num_frames} frames from {os.path.basename(video_path)} to {frames_dir}"
            results.append(result_msg)
        except Exception as e:
            error_msg = f"❌ Error processing {os.path.basename(video_path)}: {str(e)}"
            results.append(error_msg)

        if status_callback:
            await status_callback("\n".join(results))

        # Small delay to prevent UI freezing
        await asyncio.sleep(0.1)

    return "\n".join(results)

def navigate_to_parent(current_path):
    """Navigate to parent directory"""
    parent = str(Path(current_path).parent)
    videos = list_videos(parent)
    return parent, gr.update(choices=list_directories(parent), value=parent), gr.update(choices=videos, value=[])

def handle_path_input(path_input):
    """Handle path input, converting Windows paths to WSL if needed"""
    converted_path = convert_path_if_needed(path_input, "wsl" if is_wsl() else "windows")
    videos = list_videos(converted_path)
    return converted_path, gr.update(choices=list_directories(converted_path), value=converted_path), gr.update(choices=videos, value=[])

def update_current_dir(selected_dir, current):
    if not selected_dir:
        videos = list_videos(current)
        return current, gr.update(choices=list_directories(current), value=current), gr.update(choices=videos, value=[])

    # Selected_dir could be relative or absolute
    if os.path.isabs(selected_dir):
        new_path = selected_dir
    else:
        new_path = os.path.join(current, selected_dir)

    videos = list_videos(new_path)
    # Return updated current directory, list of subdirectories, and list of videos
    return new_path, gr.update(choices=list_directories(new_path), value=new_path), gr.update(choices=videos, value=[])

def select_all_videos(current_dir, current_selection):
    """Select all videos or none if all are already selected"""
    all_videos = list_videos(current_dir)

    if len(current_selection) == len(all_videos):
        return []  # Deselect
    else:
        return all_videos  # Select all

def create_frame_extractor_tab():
    """Create the frame extractor tab for the Gradio UI"""
    with gr.Tab("Video Frame Extractor") as frame_tab:
        gr.Markdown("# Video Frame Extractor")

        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("## Navigation")
                current_dir = gr.Textbox(value="/mnt/d/sync_ai/training", label="Current Directory")
                up_button = gr.Button("Go Up")
                dir_dropdown = gr.Dropdown(
                    value="/mnt/d/sync_ai/training",
                    choices=list_directories("/mnt/d/sync_ai/training"),
                    label="Select Subdirectory",
                    allow_custom_value=True
                )

                with gr.Row():
                    refresh_btn = gr.Button("Refresh")

                with gr.Column():
                    with gr.Row():
                        video_selector = gr.CheckboxGroup(choices=[], label="Select Videos")
                    with gr.Row():
                        select_all_btn = gr.Button("Select All/None", size='sm')

                output_dir = gr.Textbox(label="Output Directory (leave empty for default)")
                num_frames = gr.Slider(minimum=1, maximum=30, value=5, step=1, label="Number of frames to extract")

                process_btn = gr.Button("Extract Frames", variant="primary")

            with gr.Column(scale=3):
                output = gr.Textbox(label="Status", lines=10)

        # Update directory listing when going up
        up_button.click(
            navigate_to_parent,
            [current_dir],
            [current_dir, dir_dropdown, video_selector]
        )

        # Event for when current_dir is manually changed
        current_dir.submit(
            handle_path_input,
            [current_dir],
            [current_dir, dir_dropdown, video_selector]
        )

        # Change refreshes both directory and video lists
        dir_dropdown.change(
            update_current_dir,
            [dir_dropdown, current_dir],
            [current_dir, dir_dropdown, video_selector]
        )

        # refresh updates subdirectory list
        refresh_btn.click(
            lambda x: (x, gr.update(choices=list_directories(x), value=x), gr.update(choices=list_videos(x), value=[])),
            [current_dir],
            [current_dir, dir_dropdown, video_selector]
        )

        # Select all videos button
        select_all_btn.click(
            select_all_videos,
            inputs=[current_dir, video_selector],
            outputs=video_selector
        )

        # (async) Process videos
        process_btn.click(
            fn=process_videos,
            inputs=[current_dir, video_selector, output_dir, num_frames],
            outputs=output,
            api_name="process_videos"
        )

    return {"tab": frame_tab}
