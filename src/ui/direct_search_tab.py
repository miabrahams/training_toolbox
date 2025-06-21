import gradio as gr
import time
from typing import Dict, Any

from src.tag_analyzer.database import TagDatabase

def direct_search(db: TagDatabase | None, query: str, limit: int = 100):
    """Search prompts directly from the database"""
    if db is None:
        return "❌ Database not initialized. Load data first.", None

    if not query or len(query) < 3:
        return "Enter at least 3 characters to search.", None

    results = db.search_positive_prompts(query=query, limit=limit)

    # Format results for the UI
    if not results:
        return f"No results found for '{query}'.", None

    # Generate markdown output
    md_output = f"# Search Results: '{query}'\n\n"
    md_output += f"Found {len(results)} matches (limit {limit})\n\n"

    # Add results
    for i, item in enumerate(results):
        md_output += f"### {i+1}. {item['positive_prompt']}\n\n"
        md_output += f"Image: `{item['file_path']}`\n\n"
        md_output += "---\n\n"

    return md_output, results

def create_direct_search_tab(db_state: gr.State) -> Dict[str, Any]:
    """
    Create the direct search tab for the Gradio UI

    Args:
        db_state: Gradio State object containing the TagDatabase instance

    Returns:
        Dict with tab components
    """
    with gr.Tab("Direct Search") as direct_search_tab:
        with gr.Row():
            with gr.Column():
                search_status = gr.Markdown("**Status**: No database loaded")
                gr.Markdown("## Direct Database Search")
                gr.Markdown("Search for prompts directly in the database. Results appear as you type.")

                with gr.Row():
                    query = gr.Textbox(
                        label="Search Query",
                        placeholder="Enter search term (min 3 chars)...",
                        interactive=True
                    )

                with gr.Row():
                    limit = gr.Slider(
                        minimum=10,
                        maximum=500,
                        value=100,
                        step=10,
                        label="Result Limit"
                    )

        with gr.Row():
            with gr.Column(scale=2):
                search_output = gr.Markdown("Enter a search query to see results.")
            with gr.Column(scale=1):
                result_data = gr.JSON(label="Result Data")

        # State for debouncing
        last_call = gr.State(0)

        # Debouncer wrapper for search function
        def debounced_search(db, query, limit, last_call_time):
            """A debounced version of the search function"""
            current_time = time.time()
            if current_time - last_call_time < 0.5:
                # Not enough time passed, so we skip the update
                return gr.skip(), gr.skip(), last_call_time

            # Run the search
            output, data = direct_search(db, query, limit)
            return output, data, current_time

        # Function to update the UI when database is loaded
        def update_ui_on_load(db):
            if db is None:
                return "**Status**: No database loaded", gr.update(interactive=False)

            status_text = "**Status**: Database loaded. Ready to search."
            return status_text, gr.update(interactive=True)

        db_state.change(
            fn=update_ui_on_load,
            inputs=[db_state],
            outputs=[search_status, query]
        )

        # Connect search query to search function with debouncing
        query.change(
            fn=debounced_search,
            inputs=[db_state, query, limit, last_call],
            outputs=[search_output, result_data, last_call],
            show_progress="hidden"
        )

    return {"tab": direct_search_tab}