import random
import gradio as gr
import numpy as np
from typing import Dict, Any

from src.tag_analyzer import TagAnalyzer, create_analyzer
from pathlib import Path

# Global analyzer instance
analyzer: TagAnalyzer | None = None

def initialize_analyzer(db_path: str, data_dir: str, force_recompute=False, progress=gr.Progress()):
    """Initialize the analyzer with given paths and display progress"""
    global analyzer

    analyzer = create_analyzer(Path(db_path), Path(data_dir), progress)

    if analyzer is not None and analyzer.clusters is not None:
        return f"Loaded {len(analyzer.prompt_texts)} prompts with {len(np.unique(analyzer.clusters))-1} clusters."
    else:
        return "Error: Failed to initialize the tag analyzer. Check the database path."

def get_cluster_summary(sample_size=5, screen_dirs=None, show_paths=False, progress=gr.Progress()):
    """Get cluster summary for Gradio UI"""
    global analyzer
    if analyzer is None:
        return "Error: Analyzer not initialized. Please initialize first."

    def progress_callback(progress_value, status_text):
        progress(progress_value, status_text)

    result = analyzer.get_cluster_summary(
        sample_size=sample_size,
        screen_dirs=screen_dirs.split(",") if screen_dirs and screen_dirs.strip() else None,
        show_paths=show_paths,
        progress=progress_callback
    )

    # Format the results for the UI
    if "error" in result:
        return result["error"]

    # Generate detailed markdown output
    md_output = "# Cluster Summary\n\n"
    md_output += "## Statistics\n\n"
    md_output += f"- Total clusters: {result['stats']['total_clusters']}\n"
    md_output += f"- Displayed clusters: {result['stats']['displayed_clusters']}\n"
    md_output += f"- Noise points: {result['stats']['noise_points']}\n"
    md_output += f"- Total prompts: {result['stats']['total_prompts']}\n"

    if screen_dirs:
        md_output += f"- Screened directories: {screen_dirs}\n"
        md_output += f"- Clusters filtered out: {result['stats']['screened_clusters']}\n"

    # Add cluster details
    md_output += "\n## Clusters\n\n"

    for summary in result['summaries']:
        md_output += f"### Cluster {summary['cluster_id']} - {summary['size']} prompts\n\n"
        md_output += f"**Common tokens:** {', '.join(summary['common_tokens'])}\n\n"
        md_output += f"**Representative prompt:** {summary['representative']}\n\n"

        if show_paths and 'image_path' in summary and summary['image_path']:
            md_output += f"**Image:** {summary['image_path']}\n\n"

        md_output += "**Sample prompts:**\n\n"
        for i, prompt in enumerate(summary.get('samples', [])[:5]):
            md_output += f"{i+1}. {prompt}\n"

        md_output += "\n---\n\n"

    return md_output

def analyze_directory(directory, sample_size=5, noise_sample=10, progress=gr.Progress()):
    """Analyze directory contributions to clusters for Gradio UI"""
    global analyzer
    if analyzer is None:
        return "Error: Analyzer not initialized. Please initialize first."

    def progress_callback(progress_value, status_text):
        progress(progress_value, status_text)

    result = analyzer.analyze_directory(
        directory=directory,
        sample_size=sample_size,
        noise_sample=noise_sample,
        progress=progress_callback
    )

    # Format the results for the UI
    if "error" in result:
        return result["error"]

    # Generate detailed markdown output
    md_output = f"# Directory Analysis: {result['directory']}\n\n"
    md_output += f"## Statistics\n\n"
    md_output += f"- Total images: {result['stats']['total_images']}\n"
    md_output += f"- Clustered images: {result['stats']['clustered_images']}\n"
    md_output += f"- Noise images: {result['stats']['noise_images']}\n"
    md_output += f"- Cluster count: {result['stats']['cluster_count']}\n\n"

    # Add cluster distribution
    md_output += "## Cluster Distribution\n\n"

    for cluster_id, data in sorted(
        [(int(k), v) for k, v in result['clusters'].items()],
        key=lambda x: x[1]['count'],
        reverse=True
    ):
        md_output += f"- Cluster {cluster_id}: {data['count']} images\n"

    # Add sample prompts for each cluster
    md_output += "\n## Cluster Samples\n\n"

    for cluster_id, data in sorted(
        [(int(k), v) for k, v in result['clusters'].items()],
        key=lambda x: int(x[0])
    ):
        md_output += f"### Cluster {cluster_id} ({data['count']} prompts)\n\n"
        for i, prompt in enumerate(data['samples']):
            md_output += f"{i+1}. {prompt}\n"
        md_output += "\n"

    # Add noise samples if available
    if result["noise_samples"]:
        md_output += "## Noise Cluster Samples\n\n"
        for i, prompt in enumerate(result["noise_samples"]):
            md_output += f"{i+1}. {prompt}\n"

    return md_output

def analyze_tags(top_n=20, include_noise=False, cluster_pairs=5, sample_size=10, progress=gr.Progress()):
    """Analyze tag distribution for Gradio UI"""
    global analyzer
    if analyzer is None:
        return "Error: Analyzer not initialized. Please initialize first."

    def progress_callback(progress_value, status_text):
        progress(progress_value, status_text)

    result = analyzer.analyze_tags(
        top_n=top_n,
        include_noise=include_noise,
        cluster_pairs=cluster_pairs,
        sample_size=sample_size,
        progress=progress_callback
    )

    # Format the results for the UI
    if "error" in result:
        return result["error"]

    # Generate detailed markdown output
    md_output = "# Tag Distribution Analysis\n\n"

    # Overall tag distribution
    md_output += f"## Top {len(result['overall_tags'])} Tags Overall\n\n"
    for tag, count in result['overall_tags'].items():
        md_output += f"- {tag}: {count}\n"

    # Per-cluster tag distribution
    md_output += "\n## Top Tags by Cluster\n\n"
    for cluster_id, tags in sorted(result['cluster_tags'].items(), key=lambda x: int(x[0])):
        md_output += f"### Cluster {cluster_id}\n\n"
        for tag, count in tags.items():
            md_output += f"- {tag}: {count}\n"
        md_output += "\n"

    # Cluster pair differences
    if result['pair_differences']:
        md_output += "\n## Cluster Pair Differences\n\n"
        for pair_key, pair_data in result['pair_differences'].items():
            cluster_a, cluster_b = pair_data['clusters']
            md_output += f"### Cluster {cluster_a} vs Cluster {cluster_b}\n\n"
            for tag, count in pair_data['differences'].items():
                md_output += f"- {tag}: {count}\n"
            md_output += "\n"

    return md_output

def analyze_modifiers(top_n=50, sample_size=20, max_clusters=None,
                      show_examples=False, max_examples=3, progress=gr.Progress()):
    """Analyze common modifiers for Gradio UI"""
    global analyzer
    if analyzer is None:
        return "Error: Analyzer not initialized. Please initialize first."

    if max_clusters is not None and max_clusters <= 0:
        max_clusters = None

    def progress_callback(progress_value, status_text):
        progress(progress_value, status_text)

    result = analyzer.analyze_modifiers(
        top_n=top_n,
        sample_size=sample_size,
        max_clusters=max_clusters,
        show_examples=show_examples,
        max_examples=max_examples,
        progress=progress_callback
    )

    # Format the results for the UI
    if "error" in result:
        return result["error"]

    # Generate detailed markdown output
    md_output = f"# Common Tag Modifier Analysis\n\n"
    md_output += f"## Top {len(result['modifiers'])} Tag Modifiers\n\n"

    for modifier, data in result['modifiers'].items():
        md_output += f"### {modifier}: {data['count']} occurrences\n\n"

        # If examples are included
        if "examples" in data:
            md_output += "Examples:\n\n"
            for i, example in enumerate(data["examples"]):
                # Truncate examples to keep output manageable
                if len(example) > 100:
                    example = example[:100] + "..."
                md_output += f"{i+1}. {example}\n"
            md_output += "\n"

    return md_output

def generate_visualization(sample_size=100, directory=None, with_diffs=False, progress=gr.Progress()):
    """Generate cluster visualization for Gradio UI"""
    global analyzer
    if analyzer is None:
        return None, "Error: Analyzer not initialized. Please initialize first."

    def progress_callback(progress_value, status_text):
        progress(progress_value, status_text)

    # Use the analyzer's generate_plot method which returns the plot image and text
    plot_img, text_output = analyzer.generate_plot(
        sample_size=sample_size,
        directory=directory if directory and directory.strip() else None,
        with_diffs=with_diffs,
        progress_callback=progress_callback
    )

    return plot_img, text_output

# Functions for the more detailed UI components
def generate_visualization_fn(analyzer_instance, sample_size, directory, with_diffs):
    """Generate visualization with more detailed return data for UI"""
    if not analyzer_instance:
        return None, "### Error\n\nPlease load data first.", None, None

    try:
        result = analyzer_instance.generate_visualization(
            sample_size=sample_size,
            directory=directory if directory else None,
            with_diffs=with_diffs
        )

        if "error" in result:
            return None, f"### Error\n\n{result['error']}", None, None

        # Create matplotlib figure from the result data
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(10, 8))

        # Get data from points
        x = [p["x"] for p in result["points"]]
        y = [p["y"] for p in result["points"]]
        clusters = [p["cluster"] for p in result["points"]]

        # Plot points
        scatter = plt.scatter(x, y, c=clusters, cmap='tab20', alpha=0.6, s=10)
        plt.colorbar(scatter, label="Cluster")
        plt.title('Prompt Clusters')
        plt.xlabel('UMAP Dimension 1')
        plt.ylabel('UMAP Dimension 2')

        # Get cluster samples for display
        cluster_samples_dict = {}
        for cluster_id, data in result["cluster_samples"].items():
            cluster_samples_dict[cluster_id] = {
                "size": data["size"],
                "common_tokens": data["common_tokens"],
                "sample": random.choice(data["samples"]) # One sample per cluster
            }

        # Create stats object
        vis_stats_dict = {
            "total_clusters": result["total_clusters"],
            "total_points": result["total_points"],
            "noise_points": result["noise_points"]
        }

        return fig, None, cluster_samples_dict, vis_stats_dict
    except Exception as e:
        return None, f"### Error\n\n{str(e)}", None, None

def generate_summary(analyzer_instance, sample_size, screen_dirs_str, show_paths):
    """Generate summary with more detailed return data for UI"""
    if not analyzer_instance:
        return "### Error\n\nPlease load data first.", {"error": "No data loaded"}

    # Process screen directories
    screen_dirs = None
    if screen_dirs_str:
        screen_dirs = [d.strip() for d in screen_dirs_str.split(',') if d.strip()]

    try:
        result = analyzer_instance.get_cluster_summary(
            sample_size=sample_size,
            screen_dirs=screen_dirs,
            show_paths=show_paths
        )

        # Format output for Markdown
        md_output = "### Cluster Summary\n\n"
        for summary in result['summaries']:
            md_output += f"#### Cluster {summary['cluster_id']} - {summary['size']} prompts\n\n"
            md_output += f"Common tokens: {', '.join(summary['common_tokens'])}\n\n"
            md_output += f"Representative prompt: {summary['representative']}\n\n"
            if show_paths and 'image_path' in summary and summary['image_path']:
                md_output += f"Representative image: {summary['image_path']}\n\n"
            md_output += "---\n\n"

        return md_output, result['stats']
    except Exception as e:
        return f"### Error\n\n{str(e)}", {"error": str(e)}

def analyze_directory_fn(analyzer_instance, directory, sample_size, noise_sample):
    """Analyze directory with more detailed return data for UI"""
    if not analyzer_instance:
        return "### Error\n\nPlease load data first.", {"error": "No data loaded"}, None

    try:
        result = analyzer_instance.analyze_directory(
            directory=directory,
            sample_size=sample_size,
            noise_sample=noise_sample
        )

        # Format output for Markdown
        md_output = f"### Directory Analysis: {result['directory']}\n\n"
        md_output += f"Total images: {result['stats']['total_images']}\n\n"
        md_output += f"Clustered images: {result['stats']['clustered_images']}\n\n"
        md_output += f"Noise images: {result['stats']['noise_images']}\n\n"
        md_output += f"Contributing to {result['stats']['cluster_count']} clusters\n\n"

        # Add cluster distribution
        md_output += "#### Cluster Distribution\n\n"
        for cluster_id, data in sorted(
            [(int(k), v) for k, v in result['clusters'].items()],
            key=lambda x: x[1]['count'],
            reverse=True
        ):
            md_output += f"Cluster {cluster_id}: {data['count']} images\n\n"

        # Add noise samples if available
        if result["noise_samples"]:
            md_output += "#### Noise Samples\n\n"
            for i, prompt in enumerate(result["noise_samples"]):
                md_output += f"{i+1}. {prompt}\n\n"

        return md_output, result['stats'], result['clusters']
    except Exception as e:
        return f"### Error\n\n{str(e)}", {"error": str(e)}, None

def analyze_tags_fn(analyzer_instance, top_n, include_noise, cluster_pairs, sample_size):
    """Analyze tags with more detailed return data for UI"""
    if not analyzer_instance:
        return "### Error\n\nPlease load data first.", None, None

    try:
        result = analyzer_instance.analyze_tags(
            top_n=top_n,
            include_noise=include_noise,
            cluster_pairs=cluster_pairs,
            sample_size=sample_size
        )

        # Format output for Markdown
        md_output = "### Tag Analysis\n\n"

        # Overall tags
        md_output += "#### Overall Top Tags\n\n"
        for tag, count in result['overall_tags'].items():
            md_output += f"{tag}: {count}\n\n"

        # Per-cluster tags (summary)
        md_output += "#### Cluster Tag Summary\n\n"
        for cluster_id, tags in sorted(result['cluster_tags'].items(), key=lambda x: int(x[0])):
            md_output += f"Cluster {cluster_id}: "
            md_output += ", ".join([f"{tag} ({count})" for tag, count in tags.items()])
            md_output += "\n\n"

        # Pair differences (summary)
        if result['pair_differences']:
            md_output += "#### Cluster Pair Differences\n\n"
            for pair_key, pair_data in result['pair_differences'].items():
                cluster_a, cluster_b = pair_data['clusters']
                md_output += f"Clusters {cluster_a} vs {cluster_b}: "
                md_output += ", ".join([f"{tag} ({count})" for tag, count in pair_data['differences'].items()])
                md_output += "\n\n"

        return md_output, result['overall_tags'], result['cluster_tags']
    except Exception as e:
        return f"### Error\n\n{str(e)}", None, None

def analyze_modifiers_fn(analyzer_instance, top_n, sample_size, max_clusters, show_examples, max_examples):
    """Analyze modifiers with more detailed return data for UI"""
    if not analyzer_instance:
        return "### Error\n\nPlease load data first.", None

    try:
        result = analyzer_instance.analyze_modifiers(
            top_n=top_n,
            sample_size=sample_size,
            max_clusters=max_clusters if max_clusters else None,
            show_examples=show_examples,
            max_examples=max_examples
        )

        # Format output for Markdown
        md_output = "### Modifier Analysis\n\n"

        # List modifiers
        md_output += "#### Top Modifiers\n\n"
        for modifier, data in result['modifiers'].items():
            md_output += f"**{modifier}**: {data['count']} occurrences\n\n"

            if show_examples and "examples" in data:
                md_output += "Examples:\n\n"
                for i, example in enumerate(data["examples"]):
                    # Truncate examples to keep output manageable
                    if len(example) > 100:
                        example = example[:100] + "..."
                    md_output += f"- {example}\n\n"

        return md_output, result
    except Exception as e:
        return f"### Error\n\n{str(e)}", None

def compute_analysis(progress=gr.Progress()):
    """Compute embeddings and clusters for the analyzer"""
    def progress_callback(progress_value, status_text):
        progress(progress_value, status_text)

    try:
        # Use the passed-in analyzer to compute analysis data
        analyzer._compute_analysis_data(progress=progress_callback)

        # Return success message with stats
        clusters = analyzer.clusters if analyzer.analysis else None
        if clusters is not None:
            n_clusters = len(np.unique(clusters)) - (1 if -1 in clusters else 0)
            return f"Analysis complete: {len(analyzer.prompt_texts)} prompts, {n_clusters} clusters"
        else:
            return "Error computing analysis data"
    except Exception as e:
        return f"Error: {str(e)}"

def create_tag_analysis_tab(in_analyzer: TagAnalyzer) -> Dict[str, Any]:
    """
    Create the tag analysis tab for the Gradio UI

    Args:
        prompt_data: Initialized PromptData instance
        db: Initialized TagDatabase instance
        data_dir: Path to data directory

    Returns:
        Dict with tab components for incorporation into the main UI
    """
    global analyzer
    analyzer = in_analyzer

    # Initialize the analyzer with provided data
    with gr.Tab("Tag Analysis") as tag_tab:
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## Analysis Settings")
                # Replace the database path and data directory with just a compute button
                compute_btn = gr.Button("Compute Analysis", variant="primary")

                analysis_status = gr.Textbox(
                    label="Analysis Status",
                    value="Analysis not computed" if analyzer.analysis is None else "Analysis loaded from disk"
                )

                db_path = gr.Textbox(
                    label="Database Path",
                    value="data/prompts.sqlite",
                    info="Path to SQLite database with prompts"
                )

                data_dir = gr.Textbox(
                    label="Data Directory",
                    value="data",
                    info="Directory to save/load analysis data"
                )

                force_recompute = gr.Checkbox(
                    label="Force Recomputation",
                    value=False,
                    info="Force recomputation of embeddings and clusters"
                )

                load_btn = gr.Button("Load Data", variant="primary")

                load_status = gr.Textbox(label="Status", value="Not loaded")

            with gr.Column(scale=2):
                gr.Markdown("## Analysis Tools")

                with gr.Tabs():
                    with gr.Tab("Cluster Summary"):
                        with gr.Row():
                            summary_sample_size = gr.Slider(
                                minimum=1, maximum=20, value=5, step=1,
                                label="Sample Size"
                            )

                            show_paths = gr.Checkbox(
                                label="Show Image Paths",
                                value=False
                            )

                        screen_dirs = gr.Textbox(
                            label="Screen Directories (comma separated)",
                            placeholder="dir1,dir2,dir3",
                            info="Filter out clusters with images in these directories"
                        )

                        summary_btn = gr.Button("Generate Summary")

                        with gr.Row():
                            with gr.Column(scale=2):
                                summary_output = gr.Markdown("### Cluster Summary\n\nRun analysis to see results")

                            with gr.Column(scale=1):
                                summary_stats = gr.JSON(label="Statistics")

                    with gr.Tab("Directory Analysis"):
                        with gr.Row():
                            dir_path = gr.Textbox(
                                label="Directory Path",
                                info="Path to directory to analyze"
                            )

                            dir_sample_size = gr.Slider(
                                minimum=1, maximum=20, value=5, step=1,
                                label="Sample Size"
                            )

                            noise_sample = gr.Slider(
                                minimum=0, maximum=20, value=10, step=1,
                                label="Noise Sample Size"
                            )

                        dir_btn = gr.Button("Analyze Directory")

                        with gr.Row():
                            with gr.Column():
                                dir_output = gr.Markdown("### Directory Analysis\n\nRun analysis to see results")

                            with gr.Column():
                                dir_stats = gr.JSON(label="Statistics")
                                dir_cluster_data = gr.JSON(label="Cluster Data", visible=False)

                    with gr.Tab("Tag Distribution"):
                        with gr.Row():
                            tag_top_n = gr.Slider(
                                minimum=5, maximum=50, value=20, step=5,
                                label="Top N Tags"
                            )

                            include_noise = gr.Checkbox(
                                label="Include Noise Cluster",
                                value=False
                            )

                            cluster_pairs = gr.Slider(
                                minimum=0, maximum=10, value=5, step=1,
                                label="Cluster Pairs to Compare"
                            )

                            tag_sample_size = gr.Slider(
                                minimum=5, maximum=20, value=10, step=1,
                                label="Sample Size"
                            )

                        tag_btn = gr.Button("Analyze Tags")

                        with gr.Row():
                            with gr.Column():
                                tag_output = gr.Markdown("### Tag Analysis\n\nRun analysis to see results")

                            with gr.Column():
                                overall_tags = gr.JSON(label="Overall Tags")
                                cluster_tags = gr.JSON(label="Cluster Tags", visible=False)

                    with gr.Tab("Modifier Analysis"):
                        with gr.Row():
                            mod_top_n = gr.Slider(
                                minimum=10, maximum=100, value=50, step=10,
                                label="Top N Modifiers"
                            )

                            mod_sample_size = gr.Slider(
                                minimum=5, maximum=30, value=20, step=5,
                                label="Sample Size"
                            )

                            max_clusters = gr.Number(
                                label="Max Clusters (blank for all)",
                                value=None
                            )

                        with gr.Row():
                            show_examples = gr.Checkbox(
                                label="Show Examples",
                                value=False
                            )

                            max_examples = gr.Slider(
                                minimum=1, maximum=10, value=3, step=1,
                                label="Max Examples"
                            )

                        mod_btn = gr.Button("Analyze Modifiers")

                        with gr.Row():
                            mod_output = gr.Markdown("### Modifier Analysis\n\nRun analysis to see results")
                            mod_data = gr.JSON(label="Modifier Data")

                    with gr.Tab("Visualization"):
                        with gr.Row():
                            vis_sample_size = gr.Slider(
                                minimum=50, maximum=500, value=100, step=50,
                                label="Sample Size"
                            )

                            vis_directory = gr.Textbox(
                                label="Filter Directory (optional)",
                                info="Only visualize prompts from this directory"
                            )

                            with_diffs = gr.Checkbox(
                                label="Include Differences",
                                value=False
                            )

                        vis_btn = gr.Button("Generate Visualization")

                        with gr.Row():
                            plot_output = gr.Plot(label="Cluster Visualization")

                        with gr.Row():
                            cluster_samples = gr.JSON(label="Cluster Samples")
                            vis_stats = gr.JSON(label="Visualization Statistics")

    # Initialize analyzer state (will be set when loading data)
    analyzer_state = gr.State(None)

    # Connect compute button to compute_analysis function
    compute_btn.click(
        fn=compute_analysis,
        inputs=[],
        outputs=[analysis_status]
    )

    # Define UI logic
    load_btn.click(
        fn=initialize_analyzer,
        inputs=[db_path, data_dir, force_recompute],
        outputs=[load_status]
    )

    # Connect UI to analyzer methods
    summary_btn.click(
        fn=get_cluster_summary,
        inputs=[summary_sample_size, screen_dirs, show_paths],
        outputs=[summary_output]
    )

    dir_btn.click(
        fn=analyze_directory,
        inputs=[dir_path, dir_sample_size, noise_sample],
        outputs=[dir_output]
    )

    tag_btn.click(
        fn=analyze_tags,
        inputs=[tag_top_n, include_noise, cluster_pairs, tag_sample_size],
        outputs=[tag_output]
    )

    mod_btn.click(
        fn=analyze_modifiers,
        inputs=[mod_top_n, mod_sample_size, max_clusters, show_examples, max_examples],
        outputs=[mod_output]
    )

    vis_btn.click(
        fn=generate_visualization,
        inputs=[vis_sample_size, vis_directory, with_diffs],
        outputs=[plot_output, vis_stats]
    )

    # Connect detailed UI components
    summary_btn.click(
        fn=lambda: generate_summary(analyzer, summary_sample_size.value, screen_dirs.value, show_paths.value),
        inputs=[],
        outputs=[summary_output, summary_stats]
    )

    dir_btn.click(
        fn=lambda: analyze_directory_fn(analyzer, dir_path.value, dir_sample_size.value, noise_sample.value),
        inputs=[],
        outputs=[dir_output, dir_stats, dir_cluster_data]
    )

    tag_btn.click(
        fn=lambda: analyze_tags_fn(analyzer, tag_top_n.value, include_noise.value, cluster_pairs.value, tag_sample_size.value),
        inputs=[],
        outputs=[tag_output, overall_tags, cluster_tags]
    )

    mod_btn.click(
        fn=lambda: analyze_modifiers_fn(analyzer, mod_top_n.value, mod_sample_size.value, max_clusters.value, show_examples.value, max_examples.value),
        inputs=[],
        outputs=[mod_output, mod_data]
    )

    vis_btn.click(
        fn=lambda: generate_visualization_fn(analyzer, vis_sample_size.value, vis_directory.value, with_diffs.value),
        inputs=[],
        outputs=[plot_output, vis_stats, cluster_samples, vis_stats]
    )

    return {"tab": tag_tab}