import gradio as gr
from typing import Dict, Any

from src.tag_analyzer import TagAnalyzer

def create_prompt_search_tab(analyzer_state: gr.State) -> Dict[str, Any]:
    """
    Create the prompt search tab for the Gradio UI

    Returns:
        Dict with tab components for incorporation into the main UI
    """

    def search_prompts(query, case_sensitive=False, limit=500, progress=gr.Progress()):
        """Search prompts using the passed analyzer"""
        def progress_callback(progress_value, status_text):
            progress(progress_value, status_text)

        if analyzer_state.value is None:
            return "❌ Analyzer not initialized. Load data first.", None
        analyzer: TagAnalyzer = analyzer_state.value

        result = analyzer.search_prompts(
            query=query,
            case_sensitive=case_sensitive,
            limit=limit,
            progress=progress_callback
        )

        # Format results for the UI
        if "error" in result:
            return result["error"], None

        # Generate markdown output
        md_output = f"# Search Results: '{result['query']}'\n\n"
        md_output += f"Found {result['total_matches']} matches"

        if result['limit_applied']:
            md_output += f" (showing first {result['limit']})"
        md_output += "\n\n"

        # Add results
        for i, item in enumerate(result['results']):
            md_output += f"### {i+1}. "

            # Add cluster info if available
            if item['cluster'] is not None:
                md_output += f"[Cluster {item['cluster']}] "

            # Add prompt
            md_output += f"{item['prompt']}\n\n"

            # Add image path if available
            if item['image_path']:
                md_output += f"Image: `{item['image_path']}`\n\n"

            md_output += "---\n\n"

        return md_output, result

    with gr.Tab("Prompt Search") as search_tab:
        with gr.Row():
            with gr.Column():
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

        search_btn.click(
            fn=search_prompts,
            inputs=[query, case_sensitive, limit],
            outputs=[search_output, result_data]
        )

    return {"tab": search_tab}