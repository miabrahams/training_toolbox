from collections import Counter
from typing import List, Tuple, Dict, Set, Optional
import pprint
import random
import json
import sqlite3
import matplotlib.pyplot as plt
import numpy as np
import difflib
import umap
import hdbscan
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import argparse
from typing import Any
import os
import pickle

import sys
sys.path.append('.')
from lib.comfy_analysis import ComfyImage, extract_positive_prompt
from lib.prompt_parser import clean_prompt

pp = pprint.PrettyPrinter(width=240)


def generate_embeddings(texts: List[str], model_name: str = 'all-mpnet-base-v2') -> np.ndarray:
    """Generate embeddings for a list of text prompts."""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True)
    return normalize(embeddings)

def reduce_dimensions(embeddings: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Reduce dimensionality of embeddings using UMAP."""
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=15,
        min_dist=0.1,
        metric='cosine'
    )
    return reducer.fit_transform(embeddings)

def cluster_embeddings(embeddings: np.ndarray, min_cluster_size: int = 5) -> np.ndarray:
    """Cluster embeddings using HDBSCAN."""
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric='euclidean',
        cluster_selection_method='eom'
    )
    return clusterer.fit_predict(embeddings)

def save_analysis_data(embeddings: np.ndarray, reduced_embeddings: np.ndarray,
                      clusters: np.ndarray, prompt_texts: List[str],
                      data_dir: str = '../data'):
    """Save embeddings, reduced embeddings, and clusters to disk."""
    os.makedirs(data_dir, exist_ok=True)

    # Save the data
    np.save(os.path.join(data_dir, 'embeddings.npy'), embeddings)
    np.save(os.path.join(data_dir, 'reduced_embeddings.npy'), reduced_embeddings)
    np.save(os.path.join(data_dir, 'clusters.npy'), clusters)

    # Save prompt texts to match with embeddings
    with open(os.path.join(data_dir, 'prompt_texts.pkl'), 'wb') as f:
        pickle.dump(prompt_texts, f)

    print(f"Analysis data saved to {data_dir}")

def load_analysis_data(data_dir: str = '../data') -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]] | None:
    """Load embeddings, reduced embeddings, and clusters from disk if available."""
    embedding_path = os.path.join(data_dir, 'embeddings.npy')
    reduced_path = os.path.join(data_dir, 'reduced_embeddings.npy')
    clusters_path = os.path.join(data_dir, 'clusters.npy')
    prompts_path = os.path.join(data_dir, 'prompt_texts.pkl')

    # Check if all files exist
    if all(os.path.exists(p) for p in [embedding_path, reduced_path, clusters_path, prompts_path]):
        print("Loading existing analysis data...")
        embeddings = np.load(embedding_path)
        reduced_embeddings = np.load(reduced_path)
        clusters = np.load(clusters_path)

        with open(prompts_path, 'rb') as f:
            prompt_texts = pickle.load(f)

        return embeddings, reduced_embeddings, clusters, prompt_texts

    return None

def analyze_prompts(prompt_texts: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate embeddings, reduce dimensions, and cluster the prompts."""
    # Generate embeddings
    pp.pprint("Generating embeddings...")
    embeddings = generate_embeddings(prompt_texts)

    # Reduce dimensions for visualization
    pp.pprint("Reducing dimensions...")
    reduced_embeddings = reduce_dimensions(embeddings)

    # Cluster the embeddings
    pp.pprint("Generating clusters...")
    clusters = cluster_embeddings(embeddings)

    return embeddings, reduced_embeddings, clusters

def get_representative_prompt(prompts: List[str]) -> str:
    """Select a representative prompt for a cluster at random."""
    return random.choice(prompts) if prompts else ""

def get_image_paths_from_db(conn: sqlite3.Connection) -> Dict[str, str]:
    """Extract mapping of prompt to file path from database."""
    image_paths = {}
    for row in conn.execute('SELECT file_path, prompt FROM prompts'):
        try:
            prompt = json.loads(row[1])
            filename = row[0]
            positive = extract_positive_prompt(prompt)
            image_paths[clean_prompt(positive)] = filename
        except Exception:
            pass
    return image_paths

def identify_screened_clusters(
    clusters: np.ndarray,
    prompt_texts: List[str],
    image_paths: Dict[str, str],
    screen_dirs: List[str]
) -> Set[int]:
    """
    Identify clusters that have at least one image in the screen directories.

    Args:
        clusters: Cluster assignments for each prompt
        prompt_texts: List of prompt texts
        image_paths: Mapping from prompt to image path
        screen_dirs: List of directories to screen against

    Returns:
        Set of cluster IDs that are represented in screen directories
    """
    normalized_screen_dirs = [os.path.normpath(d) for d in screen_dirs]
    screened_clusters = set()

    # Check each prompt
    for idx, prompt in enumerate(prompt_texts):
        cluster_id = clusters[idx]

        # Skip noise cluster
        if cluster_id == -1:
            continue

        # Check if this prompt's image is in a screened directory
        image_path = image_paths.get(prompt)
        if image_path:
            # Check if this image is in any of the screened directories
            for screen_dir in normalized_screen_dirs:
                if os.path.normpath(image_path).startswith(screen_dir):
                    screened_clusters.add(cluster_id)
                    break

    return screened_clusters

def make_cluster_summaries(clusters: np.ndarray, prompts: Dict[str, int],
                           sample_size: int = 5,
                           image_paths: Optional[Dict[str, str]] = None,
                           screen_dirs: Optional[List[str]] = None):
    """Print a summary of each cluster with a representative prompt."""
    unique_clusters = np.unique(clusters)
    cluster_summaries = []

    # Identify which clusters to screen out if screening is requested
    screened_clusters = set()
    if screen_dirs and image_paths:
        prompt_texts = list(prompts.keys())
        screened_clusters = identify_screened_clusters(clusters, prompt_texts, image_paths, screen_dirs)
        print(f"Found {len(screened_clusters)} clusters represented in screened directories")

    for cluster in unique_clusters:
        if cluster == -1:
            continue  # Skip noise cluster

        # Skip this cluster if it's in the screened set
        if screen_dirs and cluster in screened_clusters:
            continue

        cluster_indices = np.where(clusters == cluster)[0]
        cluster_prompts = [list(prompts.keys())[i] for i in cluster_indices]

        # Select a representative prompt
        representative = get_representative_prompt(cluster_prompts)

        # Get sample prompts for analysis
        sample_indices = random.sample(list(cluster_indices), min(sample_size, len(cluster_indices)))
        sample_prompts = [list(prompts.keys())[i] for i in sample_indices]

        # Get common tokens
        common = common_tokens(sample_prompts)

        # Find representative image path if available
        representative_image = None
        if image_paths:
            representative_image = image_paths.get(representative)

        cluster_summaries.append({
            'cluster_id': cluster,
            'size': len(cluster_indices),
            'representative': representative,
            'common_tokens': common[:10] if len(common) > 10 else common,  # Limit to top 10 common tokens
            'image_path': representative_image
        })

    return cluster_summaries

def print_cluster_summary(cluster_summaries: List[Dict[str, Any]], show_image_paths: bool = False):
    """Print the summaries of clusters."""
    print("\n=== CLUSTER SUMMARY ===")
    print(f"Displaying {len(cluster_summaries)} clusters")

    for summary in cluster_summaries:
        print(f"\nCluster {summary['cluster_id']} - {summary['size']} prompts")
        print(f"Common tokens: {', '.join(summary['common_tokens'])}")
        print(f"Representative prompt: {summary['representative']}")
        if show_image_paths and 'image_path' in summary and summary['image_path']:
            print(f"Representative image: {summary['image_path']}")

def visualize_clusters(reduced_embeddings: np.ndarray, clusters: np.ndarray,
                      prompts: Dict[str, int], sample_size: int = 100):
    """Visualize the clusters and sample prompts from each."""
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1],
                         c=clusters, cmap='tab20', alpha=0.6)
    plt.colorbar(scatter)
    plt.title('Prompt Clusters')
    plt.show()

    # Print sample prompts from each cluster
    unique_clusters = np.unique(clusters)
    for cluster in unique_clusters:
        if cluster == -1:
            continue  # Skip noise cluster
        # Sample prompts
        cluster_indices = np.where(clusters == cluster)[0]
        sample_indices = random.sample(list(cluster_indices), min(sample_size, len(cluster_indices)))
        sample_prompts = [list(prompts.keys())[i] for i in sample_indices]
        print(f"\nCluster {cluster} with {cluster_indices.sum(None)} prompts:")
        pp.pprint(sample_prompts)


def common_tokens(prompts: List[str], delimiter: str = ',') -> List[str]:
    """Compute the intersection of tokens across all prompts in the cluster."""
    token_lists = [set([token.strip() for token in prompt.split(delimiter) if token.strip()]) for prompt in prompts]
    common = set.intersection(*token_lists) if token_lists else set()
    return sorted(common)

def prompt_diffs(prompt: str, baseline: str) -> str:
    """
    Use difflib to return a unified diff between the baseline prompt and the given prompt.
    """
    baseline_tokens = baseline.split()
    prompt_tokens = prompt.split()
    diff = difflib.ndiff(baseline_tokens, prompt_tokens)
    # Filter out unchanged tokens for clarity
    diff_result = ' '.join(token for token in diff if token.startswith('+') or token.startswith('-'))
    return diff_result

def analyze_cluster_diffs(cluster_prompts: List[str]):
    """
    Given a list of prompts in a cluster, print the common tokens and
    show the diff of each prompt compared to the first prompt.
    """
    if not cluster_prompts:
        return

    print("\n--- Diff Analysis for Cluster ---")
    # Compute token intersection with comma as delimiter
    common = common_tokens(cluster_prompts)
    print("Common tokens (by comma tokenization):", common)

    baseline = cluster_prompts[0]
    print("\nComparing each prompt to baseline:")
    for i, prompt in enumerate(cluster_prompts):
        diff = prompt_diffs(prompt, baseline)
        print(f"\nPrompt {i} diff:")
        print(diff)

# Example usage within visualize_clusters, iterating over clusters:
def visualize_clusters_with_diffs(reduced_embeddings: np.ndarray, clusters: np.ndarray,
                      prompts: Dict[str, int], sample_size: int = 100):
    """Visualize clusters; also run diff analysis on a sampled prompt group per cluster."""
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1],
                         c=clusters, cmap='tab20', alpha=0.6)
    plt.colorbar(scatter)
    plt.title('Prompt Clusters')
    plt.show()

    unique_clusters = np.unique(clusters)
    for cluster in unique_clusters:
        if cluster == -1:
            continue  # Skip noise cluster
        cluster_indices = np.where(clusters == cluster)[0]
        sample_indices = random.sample(list(cluster_indices), min(sample_size, len(cluster_indices)))
        sample_prompts = [list(prompts.keys())[i] for i in sample_indices]
        print(f"\nCluster {cluster} ({len(cluster_indices)} prompts):")
        for prompt in sample_prompts[0:3]:  # show a few samples
            print(prompt)

        # Run common diff analysis on these sample prompts
        analyze_cluster_diffs(sample_prompts[:5])  # analyze first 5 prompts in the cluster

def analyze_directory_clusters(
    directory_path: str,
    clusters: np.ndarray,
    prompt_texts: List[str],
    image_paths: Dict[str, str]
) -> Dict[int, List[str]]:
    """
    Analyze how a specific directory contributes to clusters.

    Args:
        directory_path: Path to the directory to analyze
        clusters: Cluster assignments for each prompt
        prompt_texts: List of prompt texts
        image_paths: Mapping from prompt to image path

    Returns:
        Dictionary mapping cluster IDs to lists of prompts in the specified directory
    """
    directory_path = os.path.normpath(directory_path)
    directory_clusters = {}

    for idx, prompt in enumerate(prompt_texts):
        cluster_id = clusters[idx]
        image_path = image_paths.get(prompt)

        if image_path and os.path.normpath(image_path).startswith(directory_path):
            if cluster_id not in directory_clusters:
                directory_clusters[cluster_id] = []
            directory_clusters[cluster_id].append(prompt)

    return directory_clusters

def print_noise_cluster_prompts(clusters: np.ndarray, prompt_texts: List[str]):
    """Print the prompts that are in the noise cluster."""
    noise_indices = np.where(clusters == -1)[0]
    noise_prompts = [prompt_texts[i] for i in noise_indices]

    print("\n=== NOISE CLUSTER PROMPTS ===")
    for prompt in noise_prompts:
        print(prompt)

def print_directory_noise_cluster(
    directory_path: str,
    clusters: np.ndarray,
    prompt_texts: List[str],
    image_paths: Dict[str, str],
    max_samples: int = 10
):
    """
    Print a sample of prompts from the noise cluster that are in the specified directory.

    Args:
        directory_path: Path to the directory to analyze
        clusters: Cluster assignments for each prompt
        prompt_texts: List of prompt texts
        image_paths: Mapping from prompt to image path
        max_samples: Maximum number of noise samples to display
    """
    directory_path = os.path.normpath(directory_path)
    noise_prompts = []

    # Find all noise cluster prompts from the specified directory
    for idx, prompt in enumerate(prompt_texts):
        if clusters[idx] == -1:  # Noise cluster
            image_path = image_paths.get(prompt)
            if image_path and os.path.normpath(image_path).startswith(directory_path):
                noise_prompts.append(prompt)

    # Sample and print
    print(f"\n=== NOISE CLUSTER PROMPTS FROM {directory_path} ===")
    print(f"Found {len(noise_prompts)} prompts in the noise cluster from this directory")

    if noise_prompts:
        samples = random.sample(noise_prompts, min(max_samples, len(noise_prompts)))
        for i, prompt in enumerate(samples):
            print(f"\n{i+1}. {prompt}")
    else:
        print("No noise cluster prompts found in this directory.")

# Refactored main() into smaller functions
def load_or_compute_analysis_data(args, conn):
    """Load existing analysis data or compute new data if needed."""
    # Try to load existing analysis data first if not forcing recomputation
    if not args.force_recompute:
        analysis_data = load_analysis_data(args.data_dir)
        if analysis_data is not None:
            embeddings, reduced_embeddings, clusters, prompt_texts = analysis_data
            # Create prompts Counter from loaded prompt_texts
            prompts = Counter(prompt_texts)
            print(f"Loaded existing analysis data with {len(prompt_texts)} prompts")
            return embeddings, reduced_embeddings, clusters, prompt_texts, prompts

    # If no existing data or force recompute, process from scratch
    BadImages: List[ComfyImage] = []
    positives: List[str] = []

    print(f"Loading prompts from database: {args.db}")
    for row in conn.execute('SELECT file_path, prompt FROM prompts'):
        try:
            prompt = json.loads(row[1])
            filename = row[0]
            img = ComfyImage(filename, prompt, {})
            positive = extract_positive_prompt(prompt)
            positives.append(positive)
        except Exception:
            BadImages.append(img)
            pass

    print(f"{len(positives)} / {len(positives)+len(BadImages)}")
    if BadImages:
        print(f"Bad images: {[b.filename for b in BadImages]}")

    prompts = Counter([clean_prompt(p) for p in positives])

    # Run the analysis
    prompt_texts = list(prompts.keys())
    embeddings, reduced_embeddings, clusters = analyze_prompts(prompt_texts)

    # Save the analysis data for future use
    save_analysis_data(embeddings, reduced_embeddings, clusters, prompt_texts, args.data_dir)

    return embeddings, reduced_embeddings, clusters, prompt_texts, prompts

def run_cluster_analysis(args, clusters, prompts, prompt_texts, reduced_embeddings, image_paths):
    """Run cluster analysis and visualization based on command-line args."""
    # Print cluster summary with representative prompts
    # Pass screen directories if specified
    screen_dirs = args.screen_dir if args.screen_dir else None
    cluster_summaries = make_cluster_summaries(
        clusters, prompts, args.sample_size,
        image_paths=image_paths,
        screen_dirs=screen_dirs
    )

    print_cluster_summary(cluster_summaries, show_image_paths=args.show_paths)

    # Only display visualizations if requested
    if args.graph:
        visualize_clusters(reduced_embeddings, clusters, prompts)
        visualize_clusters_with_diffs(reduced_embeddings, clusters, prompts, args.sample_size)

    # Print some statistics
    n_clusters = len(np.unique(clusters)) - (1 if -1 in clusters else 0)
    noise_points = np.sum(clusters == -1)
    total_displayed = len(cluster_summaries)

    print("\nAnalysis Results:")
    print(f"Number of clusters: {n_clusters}")
    print(f"Clusters displayed: {total_displayed}")
    if screen_dirs:
        print(f"Clusters filtered out by screening: {n_clusters - total_displayed}")
        print(f"Screened directories: {screen_dirs}")
    print(f"Noise points: {noise_points}")
    print(f"Total prompts: {len(prompt_texts)}")

def analyze_specific_directory(args, clusters, prompt_texts, image_paths):
    """If specified, analyze how a specific directory contributes to clusters."""
    if not args.analyze_dir:
        return

    directory_clusters = analyze_directory_clusters(
        args.analyze_dir, clusters, prompt_texts, image_paths
    )

    print(f"\n=== CLUSTER CONTRIBUTIONS FOR DIRECTORY: {args.analyze_dir} ===")

    # Get counts by cluster
    cluster_counts = {cluster_id: len(prompts) for cluster_id, prompts in directory_clusters.items()}
    total_images = sum(cluster_counts.values())

    # Print summary statistics
    print(f"Total images in directory: {total_images}")
    print(f"Images assigned to clusters: {total_images - cluster_counts.get(-1, 0)}")
    print(f"Images in noise cluster: {cluster_counts.get(-1, 0)}")
    print(f"Directory contributes to {len(cluster_counts) - (1 if -1 in cluster_counts else 0)} clusters")

    # Print clusters (excluding noise cluster) with descending count order
    print("\nCluster distribution:")
    for cluster_id, count in sorted(
        [(k, v) for k, v in cluster_counts.items() if k != -1],
        key=lambda x: x[1],
        reverse=True
    ):
        print(f"Cluster {cluster_id}: {count} images")

    # Print sample prompts for each cluster
    max_sample = args.sample_size
    for cluster_id, prompts in sorted(directory_clusters.items()):
        if cluster_id == -1:  # Skip noise cluster, handle it separately
            continue
        print(f"\nCluster {cluster_id} ({len(prompts)} prompts):")
        for prompt in prompts[:max_sample]:
            print(f"  {prompt}")

    # Print sample of noise cluster prompts
    if args.noise_sample > 0:
        print_directory_noise_cluster(
            args.analyze_dir, clusters, prompt_texts, image_paths, args.noise_sample
        )

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Analyze and cluster AI image prompts')
    parser.add_argument('--graph', action='store_true', help='Display clustering graphs')
    parser.add_argument('--db', default='data/prompts.sqlite', help='Path to SQLite database')
    parser.add_argument('--min-cluster-size', type=int, default=5, help='Minimum cluster size')
    parser.add_argument('--sample-size', type=int, default=5, help='Sample size for cluster analysis')
    parser.add_argument('--noise-sample', type=int, default=10,
                       help='Number of noise cluster samples to display for directory analysis')
    parser.add_argument('--data-dir', default='data', help='Directory to save/load analysis data')
    parser.add_argument('--force-recompute', action='store_true',
                        help='Force recomputation of embeddings and clusters')
    parser.add_argument('--screen-dir', action='append', default=[],
                        help='Directory paths to screen against (can be specified multiple times)')
    parser.add_argument('--show-paths', action='store_true',
                        help='Show image file paths in cluster summary')
    parser.add_argument('--analyze-dir', help='Directory to analyze for cluster contributions')
    args = parser.parse_args()

    # Connect to DB for image paths
    conn = sqlite3.connect(args.db)
    image_paths = get_image_paths_from_db(conn)

    # Load or compute the analysis data
    embeddings, reduced_embeddings, clusters, prompt_texts, prompts = load_or_compute_analysis_data(args, conn)

    # Run cluster analysis and visualization
    run_cluster_analysis(args, clusters, prompts, prompt_texts, reduced_embeddings, image_paths)

    # If requested, analyze a specific directory's contributions
    analyze_specific_directory(args, clusters, prompt_texts, image_paths)

if __name__ == "__main__":
    main()