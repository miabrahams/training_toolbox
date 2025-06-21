import gradio as gr
from typing import Dict, Any
from src.tag_analyzer.types import ErrorResult

from src.tag_analyzer import TagAnalyzer

def search_prompts(analyzer: TagAnalyzer | None, query, case_sensitive=False, limit=500, progress=gr.Progress()):
    """Search prompts using the analyzer"""
    if analyzer is None:
        return "❌ Analyzer not initialized. Load data first.", None

    def progress_callback(progress_value, status_text):
        progress(progress_value, status_text)

    result = analyzer.search_prompts(
        query=query,
        case_sensitive=case_sensitive,
        limit=limit,
        progress=progress_callback
    )

    # Format results for the UI
    if isinstance(result, ErrorResult):
        return result.error, None

    # Generate markdown output
    md_output = f"# Search Results: '{result.query}'\n\n"
    md_output += f"Found {result.total_matches} matches"

    if result.limit_applied:
        md_output += f" (showing first {result.limit})"
    md_output += "\n\n"

    # Add results
    for i, item in enumerate(result.results):
        md_output += f"### {i+1}. "

        # Add cluster info if available
        if item.cluster is not None:
            md_output += f"[Cluster {item.cluster}] "

        # Add prompt
        md_output += f"{item.prompt}\n\n"

        # Add image path if available
        if item.image_path:
            md_output += f"Image: `{item.image_path}`\n\n"

        md_output += "---\n\n"

    return md_output, result

def create_prompt_search_tab(analyzer_state: gr.State) -> Dict[str, Any]:
    """
    Create the prompt search tab for the Gradio UI

    Args:
        analyzer_state: Gradio State object containing the analyzer instance

    Returns:
        Dict with tab components for incorporation into the main UI
    """
    with gr.Tab("Prompt Search") as search_tab:
        with gr.Row():
            with gr.Column():
                # Add status indicator
                search_status = gr.Markdown("**Status**: No analyzer loaded")

                gr.Markdown("## Search Prompts")

                with gr.Row():
                    query = gr.Textbox(
                        label="Search Query",
                        placeholder="Enter search term...",
                        info="Search through all prompts for this text"
                    )

                with gr.Row():
                    case_sensitive = gr.Checkbox(
                        label="Case Sensitive",
                        value=False
                    )

                    limit = gr.Slider(
                        minimum=10,
                        maximum=1000,
                        value=500,
                        step=10,
                        label="Result Limit"
                    )

                search_btn = gr.Button("Search", variant="primary")

        with gr.Row():
            with gr.Column(scale=2):
                search_output = gr.Markdown("Enter a search query and click 'Search' to see results.")

            with gr.Column(scale=1):
                result_data = gr.JSON(label="Result Data")

        # Function to update the UI when analyzer is loaded
        def update_ui_on_load(analyzer):
            """Update UI elements when analyzer is loaded"""
            if analyzer is None:
                return "**Status**: No analyzer loaded", gr.update(interactive=False)

            prompt_count = len(analyzer.prompt_texts) if hasattr(analyzer, 'prompt_texts') else 0
            status_text = f"**Status**: Analyzer loaded with {prompt_count} prompts"
            btn_update = gr.update(interactive=True)

            return status_text, btn_update

        # This lets us watch the analyzer_state value and update the UI when it changes
        analyzer_state.change(
            fn=update_ui_on_load,
            inputs=[analyzer_state],
            outputs=[search_status, search_btn]
        )

        # Connect search button
        search_btn.click(
            fn=search_prompts,
            inputs=[analyzer_state, query, case_sensitive, limit],
            outputs=[search_output, result_data]
        )

    return {"tab": search_tab, "update_fn": update_ui_on_load}