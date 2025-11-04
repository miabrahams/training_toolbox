import gradio as gr
import os
from pathlib import Path
from typing import Optional, Tuple

from lib.comfy_schemas.comfy_analysis import fileToComfyImage, extract_positive_prompt, extract_negative_prompt


def extract_prompts_from_image(image_path: Path) -> Tuple[str, str, str]:
    """
    Extract positive and negative prompts from a ComfyUI image.

    Args:
        image_path: Path to the image file

    Returns:
        Tuple of (status_message, positive_prompt, negative_prompt)
    """
    if not image_path:
        return "No image provided", "", ""

    try:
        # Convert to ComfyImage object
        comfy_image = fileToComfyImage(image_path)

        # Extract positive prompt
        try:
            positive_prompt = extract_positive_prompt(comfy_image.prompt)
        except Exception as e:
            positive_prompt = f"Error extracting positive prompt: {str(e)}"

        # Extract negative prompt
        try:
            negative_prompt = extract_negative_prompt(comfy_image.prompt)
        except Exception as e:
            negative_prompt = f"Error extracting negative prompt: {str(e)}"

        # Create status message
        filename = image_path.name
        status = f"Successfully processed: {filename}"

        return status, positive_prompt, negative_prompt

    except Exception as e:
        return f"Error processing image: {str(e)}", "", ""


def create_comfy_prompt_extractor_tab():
    """Create the ComfyUI Prompt Extractor tab"""

    with gr.TabItem("ComfyUI Prompt Extractor"):
        gr.Markdown("""
        # ComfyUI Prompt Extractor

        Drag and drop a ComfyUI-generated image to extract its positive and negative prompts.
        The image must contain ComfyUI workflow metadata.
        """)

        with gr.Row():
            with gr.Column(scale=1):
                # Image upload with drag and drop
                image_input = gr.File(
                    label="Drag & Drop ComfyUI Image",
                    file_types=[".png", ".jpg", ".jpeg"],
                    file_count="single"
                )

                # Extract button
                extract_btn = gr.Button("Extract Prompts", variant="primary")

                # Status display
                status_output = gr.Textbox(
                    label="Status",
                    value="Ready to process image...",
                    interactive=False
                )

            with gr.Column(scale=2):
                # Prompt outputs
                with gr.Group():
                    positive_output = gr.Textbox(
                        label="Positive Prompt",
                        lines=8,
                        max_lines=20,
                        placeholder="Positive prompt will appear here...",
                        show_copy_button=True
                    )

                    negative_output = gr.Textbox(
                        label="Negative Prompt",
                        lines=4,
                        max_lines=10,
                        placeholder="Negative prompt will appear here...",
                        show_copy_button=True
                    )

        # Auto-extract when image is uploaded
        def on_image_upload(file: str):
            if file is None:
                return "No image uploaded", "", ""
            return extract_prompts_from_image(Path(file))

        # Manual extract button
        def on_extract_click(file: str):
            if file is None:
                return "Please upload an image first", "", ""
            return extract_prompts_from_image(Path(file))

        # Connect events
        image_input.change(
            fn=on_image_upload,
            inputs=[image_input],
            outputs=[status_output, positive_output, negative_output]
        )

        extract_btn.click(
            fn=on_extract_click,
            inputs=[image_input],
            outputs=[status_output, positive_output, negative_output]
        )

    return {
        "image_input": image_input,
        "extract_btn": extract_btn,
        "status_output": status_output,
        "positive_output": positive_output,
        "negative_output": negative_output
    }