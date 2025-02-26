from collections import Counter
from typing import List, Tuple, Dict
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


import sys
sys.path.append('..')
from lib.comfy_analysis import ComfyImage, extract_positive_prompt
from lib.prompt_parser import clean_prompt

pp = pprint.PrettyPrinter(width=240)


# Connect to DB
sqlite_db = '/home/abrahams/training_toolbox/golang/internal/testdata/prompts.sqlite'

conn = sqlite3.connect(sqlite_db)
# Iterate over prompt strings in db

BadImages: List[ComfyImage] = []
positives: List[str] = []

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

print(BadImages[0].filename)

prompts = Counter([clean_prompt(p) for p in positives])



# Add after the existing prompts Counter code
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

def analyze_prompts(prompt_texts: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate embeddings, reduce dimensions, and cluster the prompts."""
    # Generate embeddings
    embeddings = generate_embeddings(prompt_texts)

    # Reduce dimensions for visualization
    reduced_embeddings = reduce_dimensions(embeddings)

    # Cluster the reduced embeddings
    # clusters = cluster_embeddings(reduced_embeddings)
    clusters = cluster_embeddings(embeddings)

    return embeddings, reduced_embeddings, clusters


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


# Run the analysis
prompt_texts = list(prompts.keys())
embeddings, reduced_embeddings, clusters = analyze_prompts(prompt_texts)
visualize_clusters(reduced_embeddings, clusters, prompts)

# Replace your previous visualize_clusters call with:
visualize_clusters_with_diffs(reduced_embeddings, clusters, prompts)


# Print some statistics
n_clusters = len(np.unique(clusters)) - (1 if -1 in clusters else 0)
noise_points = np.sum(clusters == -1)
print("\nAnalysis Results:")
print(f"Number of clusters: {n_clusters}")
print(f"Noise points: {noise_points}")
print(f"Total prompts: {len(prompt_texts)}")