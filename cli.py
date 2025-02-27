#!/usr/bin/env python3
import argparse
from src.tag_analyzer import create_analyzer

def main():
    # Create the top-level parser
    parser = argparse.ArgumentParser(
        description='AI Prompt Cluster Analysis Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
commands:
  summary     Show a summary of all clusters
  visualize   Visualize clusters with matplotlib
  analyze-dir Analyze how a specific directory contributes to clusters
  tags        Analyze tag distribution across clusters
  modifiers   Analyze common tag modifiers across clusters
''')

    parser.add_argument('--db', default='data/prompts.sqlite', help='Path to SQLite database')
    parser.add_argument('--data-dir', default='data', help='Directory to save/load analysis data')
    parser.add_argument('--force-recompute', action='store_true', help='Force recomputation of embeddings and clusters')

    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Create parser for the "summary" command
    parser_summary = subparsers.add_parser('summary', help='Show cluster summaries')
    parser_summary.add_argument('--sample-size', type=int, default=5, help='Sample size for cluster analysis')
    parser_summary.add_argument('--screen-dir', action='append', default=[], help='Directory paths to screen against')
    parser_summary.add_argument('--show-paths', action='store_true', help='Show image file paths in summary')

    # Create parser for the "visualize" command
    parser_visual = subparsers.add_parser('visualize', help='Visualize clusters')
    parser_visual.add_argument('--sample-size', type=int, default=100, help='Sample size for visualization')
    parser_visual.add_argument('--with-diffs', action='store_true', help='Show diff analysis in visualization')
    parser_visual.add_argument('--directory', help='Only visualize prompts from this directory')

    # Create parser for the "analyze-dir" command
    parser_analyze = subparsers.add_parser('analyze-dir', help='Analyze directory contributions')
    parser_analyze.add_argument('analyze_dir', help='Directory to analyze for cluster contributions')
    parser_analyze.add_argument('--sample-size', type=int, default=5, help='Sample size for cluster analysis')
    parser_analyze.add_argument('--noise-sample', type=int, default=10, help='Number of noise samples to show')

    # Create parser for the "tags" command
    parser_tags = subparsers.add_parser('tags', help='Analyze tag distribution')
    parser_tags.add_argument('--top-n', type=int, default=20, help='Show top N most common tags')
    parser_tags.add_argument('--include-noise', action='store_true', help='Include noise cluster in analysis')
    parser_tags.add_argument('--cluster-pairs', type=int, default=5, help='Number of random cluster pairs to analyze')
    parser_tags.add_argument('--sample-size', type=int, default=10, help='Sample size per cluster for pair analysis')

    # Create parser for the "modifiers" command
    parser_mods = subparsers.add_parser('modifiers', help='Analyze common tag modifiers')
    parser_mods.add_argument('--top-n', type=int, default=50, help='Show top N most common modifiers')
    parser_mods.add_argument('--sample-size', type=int, default=20,
                            help='Number of prompts to sample within each cluster')
    parser_mods.add_argument('--max-clusters', type=int, default=None,
                            help='Maximum number of clusters to analyze (default: all)')
    parser_mods.add_argument('--show-examples', action='store_true',
                            help='Show example prompts for each modifier')
    parser_mods.add_argument('--max-examples', type=int, default=3,
                            help='Maximum number of examples to show per modifier')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    # Create and initialize the analyzer
    analyzer = create_analyzer(
        db_path=args.db,
        data_dir=args.data_dir,
        force_recompute=args.force_recompute
    )

    # Execute the appropriate command
    if args.command == 'summary':
        result = analyzer.get_cluster_summary(
            sample_size=args.sample_size,
            screen_dirs=args.screen_dir,
            show_paths=args.show_paths
        )
        print_cluster_summary(result)

    elif args.command == 'visualize':
        import matplotlib.pyplot as plt
        result = analyzer.generate_visualization(
            sample_size=args.sample_size,
            directory=args.directory,
            with_diffs=args.with_diffs
        )
        visualize_result(result)

    elif args.command == 'analyze-dir':
        result = analyzer.analyze_directory(
            directory=args.analyze_dir,
            sample_size=args.sample_size,
            noise_sample=args.noise_sample
        )
        print_directory_analysis(result)

    elif args.command == 'tags':
        result = analyzer.analyze_tags(
            top_n=args.top_n,
            include_noise=args.include_noise,
            cluster_pairs=args.cluster_pairs,
            sample_size=args.sample_size
        )
        print_tag_analysis(result)

    elif args.command == 'modifiers':
        result = analyzer.analyze_modifiers(
            top_n=args.top_n,
            sample_size=args.sample_size,
            max_clusters=args.max_clusters,
            show_examples=args.show_examples,
            max_examples=args.max_examples
        )
        print_modifier_analysis(result)

def print_cluster_summary(result):
    """Print the cluster summary in a format similar to the original script."""
    if "error" in result:
        print(f"Error: {result['error']}")
        return

    print("\n=== CLUSTER SUMMARY ===")
    print(f"Displaying {len(result['summaries'])} clusters")

    for summary in result['summaries']:
        print(f"\nCluster {summary['cluster_id']} - {summary['size']} prompts")
        print(f"Common tokens: {', '.join(summary['common_tokens'])}")
        print(f"Representative prompt: {summary['representative']}")
        if 'image_path' in summary and summary['image_path']:
            print(f"Representative image: {summary['image_path']}")

    # Print statistics
    stats = result["stats"]
    print("\nAnalysis Results:")
    print(f"Number of clusters: {stats['total_clusters']}")
    print(f"Clusters displayed: {stats['displayed_clusters']}")
    if stats.get("screen_dirs"):
        print(f"Clusters filtered out by screening: {stats['screened_clusters']}")
        print(f"Screened directories: {stats['screen_dirs']}")
    print(f"Noise points: {stats['noise_points']}")
    print(f"Total prompts: {stats['total_prompts']}")

def visualize_result(result):
    """Visualize the clusters using matplotlib."""
    if "error" in result:
        print(f"Error: {result['error']}")
        return

    import matplotlib.pyplot as plt
    import numpy as np

    points = result["points"]
    x = [p["x"] for p in points]
    y = [p["y"] for p in points]
    clusters = [p["cluster"] for p in points]

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(x, y, c=clusters, cmap='tab20', alpha=0.6)
    plt.colorbar(scatter)
    plt.title('Prompt Clusters')

    # Print sample prompts from each cluster
    for cluster_id, data in result["cluster_samples"].items():
        print(f"\nCluster {cluster_id} ({data['size']} prompts):")
        print(f"Common tokens: {', '.join(data['common_tokens'])}")

        for i, prompt in enumerate(data["samples"]):
            print(f"{i+1}. {prompt}")

        # If diff analysis is included
        if "diffs" in data:
            print("\nDiff Analysis:")
            for diff in data["diffs"]:
                print(f"Prompt {diff['prompt_index']} diff: {diff['diff']}")

    print(f"\nTotal clusters: {result['total_clusters']}")
    print(f"Total points: {result['total_points']}")
    print(f"Noise points: {result['noise_points']}")

    plt.show()

def print_directory_analysis(result):
    """Print the directory analysis in a format similar to the original script."""
    if "error" in result:
        print(f"Error: {result['error']}")
        return

    print(f"\n=== CLUSTER CONTRIBUTIONS FOR DIRECTORY: {result['directory']} ===")

    # Print summary statistics
    stats = result["stats"]
    print(f"Total images in directory: {stats['total_images']}")
    print(f"Images assigned to clusters: {stats['clustered_images']}")
    print(f"Images in noise cluster: {stats['noise_images']}")
    print(f"Directory contributes to {stats['cluster_count']} clusters")

    # Print clusters with descending count order
    print("\nCluster distribution:")
    for cluster_id, data in sorted(
        [(int(k), v) for k, v in result['clusters'].items()],
        key=lambda x: x[1]['count'],
        reverse=True
    ):
        print(f"Cluster {cluster_id}: {data['count']} images")

    # Print sample prompts for each cluster
    for cluster_id, data in sorted(
        [(int(k), v) for k, v in result['clusters'].items()],
        key=lambda x: x[0]
    ):
        print(f"\nCluster {cluster_id} ({data['count']} prompts):")
        for prompt in data["samples"]:
            print(f"  {prompt}")

    # Print noise samples if available
    if result["noise_samples"]:
        print(f"\n=== NOISE CLUSTER PROMPTS FROM {result['directory']} ===")
        print(f"Found {stats['noise_images']} prompts in the noise cluster from this directory")
        for i, prompt in enumerate(result["noise_samples"]):
            print(f"\n{i+1}. {prompt}")

def print_tag_analysis(result):
    """Print the tag analysis in a format similar to the original script."""
    if "error" in result:
        print(f"Error: {result['error']}")
        return

    print("\n=== TAG DISTRIBUTION ANALYSIS ===")

    # Print overall tag distribution
    print(f"\nTop {len(result['overall_tags'])} tags across all prompts:")
    for tag, count in result['overall_tags'].items():
        print(f"{tag}: {count}")

    # Print per-cluster tag distribution
    print("\nTop tags per cluster:")
    for cluster_id, tags in sorted(result['cluster_tags'].items(), key=lambda x: int(x[0])):
        print(f"\nCluster {cluster_id}:")
        for tag, count in tags.items():
            print(f"  {tag}: {count}")

    # Print cluster pair differences
    if result['pair_differences']:
        print("\n=== CLUSTER PAIR TAG DIFFERENCES ===")
        for pair_key, pair_data in result['pair_differences'].items():
            cluster_a, cluster_b = pair_data['clusters']
            print(f"\nDifferences between Cluster {cluster_a} and Cluster {cluster_b}:")
            for tag, count in pair_data['differences'].items():
                print(f"  {tag}: {count}")

def print_modifier_analysis(result):
    """Print the modifier analysis in a format similar to the original script."""
    if "error" in result:
        print(f"Error: {result['error']}")
        return

    print("\n=== COMMON TAG MODIFIER ANALYSIS ===")

    print(f"\nTop {len(result['modifiers'])} tag modifiers across clusters:")
    for modifier, data in result['modifiers'].items():
        print(f"{modifier}: {data['count']} occurrences")

        # If examples are included
        if "examples" in data:
            print(f"  Examples ({len(data['examples'])}):")
            for i, example in enumerate(data["examples"]):
                # Truncate examples to keep output manageable
                if len(example) > 100:
                    example = example[:100] + "..."
                print(f"    {i+1}. {example}")
            print()

if __name__ == "__main__":
    main()
