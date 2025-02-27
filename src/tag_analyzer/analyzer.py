import os
import numpy as np
import pickle
import sqlite3
import json
import random
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Set, Optional, Any
import matplotlib.pyplot as plt
from pathlib import Path
import re
import itertools

from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import umap
import hdbscan

from .utils import (
    extract_positive_prompt, clean_prompt,
    extract_tags_from_prompts, common_tokens,
    prompt_diffs, extract_normalized_diffs
)

class TagAnalyzer:
    def __init__(self, db_path="data/prompts.sqlite", data_dir="data"):
        """Initialize analyzer with database and data storage paths"""
        # Class attributes to store loaded data
        self.embeddings = None
        self.reduced_embeddings = None
        self.clusters = None
        self.prompt_texts = None
        self.prompts = None
        self.image_paths = None
        self.db_path = db_path
        self.data_dir = data_dir

    def load_data(self, force_recompute=False):
        """Load or compute analysis data"""
        if not force_recompute:
            # Try to load existing analysis data first
            analysis_data = self._load_analysis_data()
            if analysis_data is not None:
                self.embeddings, self.reduced_embeddings, self.clusters, self.prompt_texts = analysis_data
                # Create prompts Counter from loaded prompt_texts
                self.prompts = Counter(self.prompt_texts)
                print(f"Loaded existing analysis data with {len(self.prompt_texts)} prompts")

                # Load image paths from database
                self._load_image_paths()
                return True

        # If no existing data or force recompute, process from scratch
        return self._compute_analysis_data()

    def _load_analysis_data(self):
        """Load embeddings, reduced embeddings, and clusters from disk if available."""
        embedding_path = os.path.join(self.data_dir, 'embeddings.npy')
        reduced_path = os.path.join(self.data_dir, 'reduced_embeddings.npy')
        clusters_path = os.path.join(self.data_dir, 'clusters.npy')
        prompts_path = os.path.join(self.data_dir, 'prompt_texts.pkl')

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

    def _save_analysis_data(self):
        """Save embeddings, reduced embeddings, and clusters to disk."""
        os.makedirs(self.data_dir, exist_ok=True)

        # Save the data
        np.save(os.path.join(self.data_dir, 'embeddings.npy'), self.embeddings)
        np.save(os.path.join(self.data_dir, 'reduced_embeddings.npy'), self.reduced_embeddings)
        np.save(os.path.join(self.data_dir, 'clusters.npy'), self.clusters)

        # Save prompt texts to match with embeddings
        with open(os.path.join(self.data_dir, 'prompt_texts.pkl'), 'wb') as f:
            pickle.dump(self.prompt_texts, f)

        print(f"Analysis data saved to {self.data_dir}")

    def _load_image_paths(self):
        """Extract mapping of prompt to file path from database."""
        conn = sqlite3.connect(self.db_path)
        self.image_paths = {}
        for row in conn.execute('SELECT file_path, prompt FROM prompts'):
            try:
                prompt = json.loads(row[1])
                filename = row[0]
                positive = extract_positive_prompt(prompt)
                self.image_paths[clean_prompt(positive)] = filename
            except Exception:
                pass
        conn.close()

    def _compute_analysis_data(self):
        """Process data from scratch and generate embeddings and clusters."""
        conn = sqlite3.connect(self.db_path)

        bad_images = []
        positives = []

        print(f"Loading prompts from database: {self.db_path}")
        for row in conn.execute('SELECT file_path, prompt FROM prompts'):
            try:
                prompt = json.loads(row[1])
                filename = row[0]
                positive = extract_positive_prompt(prompt)
                positives.append(positive)
            except Exception:
                bad_images.append(filename)
                pass

        print(f"{len(positives)} / {len(positives) + len(bad_images)}")
        if bad_images:
            print(f"Bad images: {bad_images[:10]}{'...' if len(bad_images) > 10 else ''}")

        self.prompts = Counter([clean_prompt(p) for p in positives])
        self.prompt_texts = list(self.prompts.keys())

        # Run the analysis
        self.embeddings, self.reduced_embeddings, self.clusters = self._analyze_prompts()

        # Save the analysis data for future use
        self._save_analysis_data()

        # Load image paths
        self._load_image_paths()

        conn.close()
        return True

    def _analyze_prompts(self):
        """Generate embeddings, reduce dimensions, and cluster the prompts."""
        # Generate embeddings
        print("Generating embeddings...")
        embeddings = self._generate_embeddings(self.prompt_texts)

        # Reduce dimensions for visualization
        print("Reducing dimensions...")
        reduced_embeddings = self._reduce_dimensions(embeddings)

        # Cluster the embeddings
        print("Generating clusters...")
        clusters = self._cluster_embeddings(embeddings)

        return embeddings, reduced_embeddings, clusters

    def _generate_embeddings(self, texts: List[str], model_name: str = 'all-mpnet-base-v2') -> np.ndarray:
        """Generate embeddings for a list of text prompts."""
        model = SentenceTransformer(model_name)
        embeddings = model.encode(texts, show_progress_bar=True)
        return normalize(embeddings)

    def _reduce_dimensions(self, embeddings: np.ndarray, n_components: int = 2) -> np.ndarray:
        """Reduce dimensionality of embeddings using UMAP."""
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=15,
            min_dist=0.1,
            metric='cosine'
        )
        return reducer.fit_transform(embeddings)

    def _cluster_embeddings(self, embeddings: np.ndarray, min_cluster_size: int = 5) -> np.ndarray:
        """Cluster embeddings using HDBSCAN."""
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric='euclidean',
            cluster_selection_method='eom'
        )
        return clusterer.fit_predict(embeddings)

    def get_cluster_summary(self, sample_size=5, screen_dirs=None, show_paths=False):
        """Generate cluster summaries"""
        if not self._check_data_loaded():
            return {"error": "Data not loaded. Call load_data() first."}

        cluster_summaries = self._make_cluster_summaries(
            self.clusters, self.prompts, sample_size,
            image_paths=self.image_paths,
            screen_dirs=screen_dirs
        )

        # Add statistics
        n_clusters = len(np.unique(self.clusters)) - (1 if -1 in self.clusters else 0)
        noise_points = np.sum(self.clusters == -1)
        total_displayed = len(cluster_summaries)

        return {
            "summaries": cluster_summaries,
            "stats": {
                "total_clusters": n_clusters,
                "displayed_clusters": total_displayed,
                "noise_points": int(noise_points),
                "total_prompts": len(self.prompt_texts),
                "screened_clusters": n_clusters - total_displayed if screen_dirs else 0,
                "screen_dirs": screen_dirs
            }
        }

    def analyze_directory(self, directory, sample_size=5, noise_sample=10):
        """Analyze directory contributions to clusters"""
        if not self._check_data_loaded():
            return {"error": "Data not loaded. Call load_data() first."}

        directory_clusters = self._analyze_directory_clusters(
            directory, self.clusters, self.prompt_texts, self.image_paths
        )

        # Get counts by cluster
        cluster_counts = {cluster_id: len(prompts) for cluster_id, prompts in directory_clusters.items()}
        total_images = sum(cluster_counts.values())

        # Get samples of noise cluster prompts if requested
        noise_samples = []
        if noise_sample > 0 and -1 in directory_clusters:
            noise_prompts = directory_clusters.get(-1, [])
            noise_samples = random.sample(noise_prompts, min(noise_sample, len(noise_prompts)))

        # Create cluster data with samples
        cluster_data = {}
        max_sample = sample_size
        for cluster_id, prompts in directory_clusters.items():
            if cluster_id == -1:  # Skip noise cluster, handle separately
                continue
            cluster_data[str(cluster_id)] = {
                "count": len(prompts),
                "samples": prompts[:max_sample]
            }

        return {
            "directory": directory,
            "stats": {
                "total_images": total_images,
                "clustered_images": total_images - cluster_counts.get(-1, 0),
                "noise_images": cluster_counts.get(-1, 0),
                "cluster_count": len(cluster_counts) - (1 if -1 in cluster_counts else 0)
            },
            "clusters": cluster_data,
            "noise_samples": noise_samples
        }

    def analyze_tags(self, top_n=20, include_noise=False, cluster_pairs=5, sample_size=10):
        """Analyze tag distribution"""
        if not self._check_data_loaded():
            return {"error": "Data not loaded. Call load_data() first."}

        # Get overall tag distribution
        all_tags = extract_tags_from_prompts(self.prompt_texts)
        most_common = all_tags.most_common(top_n)

        # Get per-cluster tag distribution
        cluster_tags = self._analyze_cluster_tag_distribution()

        cluster_tag_data = {}
        for cluster_id in sorted(cluster_tags.keys()):
            if cluster_id == -1 and not include_noise:  # Skip noise cluster unless requested
                continue
            tags = cluster_tags[cluster_id].most_common(5)  # Top 5 per cluster
            cluster_tag_data[str(cluster_id)] = {tag: count for tag, count in tags}

        # Analyze differences between random cluster pairs
        pair_diffs = {}
        if cluster_pairs > 0:
            pair_diff_data = self._analyze_cluster_pair_diffs(
                max_pairs=cluster_pairs, sample_size=sample_size
            )

            for (cluster_a, cluster_b), diff_tags in pair_diff_data.items():
                pair_key = f"{cluster_a}_vs_{cluster_b}"
                pair_diffs[pair_key] = {
                    "clusters": [int(cluster_a), int(cluster_b)],
                    "differences": {tag: count for tag, count in diff_tags.most_common(10)}
                }

        return {
            "overall_tags": {tag: count for tag, count in most_common},
            "cluster_tags": cluster_tag_data,
            "pair_differences": pair_diffs
        }

    def analyze_modifiers(self, top_n=50, sample_size=20, max_clusters=None,
                          show_examples=False, max_examples=3):
        """Analyze common modifiers across clusters"""
        if not self._check_data_loaded():
            return {"error": "Data not loaded. Call load_data() first."}

        modifiers = self._get_common_modifiers(
            sample_size=sample_size,
            max_clusters=max_clusters,
            top_n=top_n
        )

        result = {
            "modifiers": {}
        }

        for modifier, count in modifiers:
            result["modifiers"][modifier] = {
                "count": count
            }

            # If requested, show example contexts
            if show_examples:
                examples = self._analyze_modifier_context(
                    modifier,
                    max_examples=max_examples
                )
                result["modifiers"][modifier]["examples"] = examples

        return result

    def generate_visualization(self, sample_size=100, directory=None, with_diffs=False):
        """Generate visualization data"""
        if not self._check_data_loaded():
            return {"error": "Data not loaded. Call load_data() first."}

        # If directory specified, filter for images in that directory
        if directory:
            dir_prompt_indices = []
            for idx, prompt in enumerate(self.prompt_texts):
                image_path = self.image_paths.get(prompt)
                if image_path and os.path.normpath(image_path).startswith(os.path.normpath(directory)):
                    dir_prompt_indices.append(idx)

            # Create filtered arrays
            if dir_prompt_indices:
                filtered_reduced = self.reduced_embeddings[dir_prompt_indices]
                filtered_clusters = self.clusters[dir_prompt_indices]
                filtered_prompts = [self.prompt_texts[i] for i in dir_prompt_indices]

                # Get cluster samples for visualization
                return self._prepare_visualization_data(
                    filtered_reduced, filtered_clusters, filtered_prompts, sample_size, with_diffs
                )
            else:
                return {"error": f"No prompts found in directory: {directory}"}
        else:
            # Use all data
            return self._prepare_visualization_data(
                self.reduced_embeddings, self.clusters, self.prompt_texts, sample_size, with_diffs
            )

    def _prepare_visualization_data(self, reduced_embeddings, clusters, prompt_texts, sample_size=100, with_diffs=False):
        """Prepare data for visualization"""
        # Convert the embeddings to a list format that can be easily sent to JS
        points = []
        for i in range(len(reduced_embeddings)):
            points.append({
                "x": float(reduced_embeddings[i, 0]),
                "y": float(reduced_embeddings[i, 1]),
                "cluster": int(clusters[i]),
                "prompt": prompt_texts[i] if i < len(prompt_texts) else ""
            })

        # Get cluster samples
        cluster_samples = {}
        unique_clusters = np.unique(clusters)

        for cluster_id in unique_clusters:
            if cluster_id == -1:
                continue  # Skip noise cluster

            # Get indices for this cluster
            cluster_indices = np.where(clusters == cluster_id)[0]

            # Sample prompts
            sample_indices = random.sample(
                list(cluster_indices),
                min(sample_size, len(cluster_indices))
            )

            sample_prompts = [prompt_texts[i] for i in sample_indices]

            # Get common tokens
            common = common_tokens(sample_prompts)

            # Store samples
            cluster_samples[str(cluster_id)] = {
                "size": len(cluster_indices),
                "common_tokens": common[:10] if len(common) > 10 else common,
                "samples": sample_prompts[:5]  # Limit to 5 samples
            }

            # Add diff analysis if requested
            if with_diffs and sample_prompts:
                diffs = []
                baseline = sample_prompts[0]
                for i, prompt in enumerate(sample_prompts[1:5]):  # Limit to 4 diffs
                    diff = prompt_diffs(prompt, baseline)
                    diffs.append({
                        "prompt_index": i + 1,
                        "diff": diff
                    })
                cluster_samples[str(cluster_id)]["diffs"] = diffs

        return {
            "points": points,
            "cluster_samples": cluster_samples,
            "total_clusters": len(unique_clusters) - (1 if -1 in unique_clusters else 0),
            "total_points": len(points),
            "noise_points": int(np.sum(clusters == -1))
        }

    def _check_data_loaded(self):
        """Check if data is loaded, return False if not"""
        return (self.embeddings is not None and
                self.clusters is not None and
                self.prompt_texts is not None)

    def _make_cluster_summaries(self, clusters, prompts, sample_size=5,
                               image_paths=None, screen_dirs=None):
        """Generate summaries for each cluster with representative prompts."""
        unique_clusters = np.unique(clusters)
        cluster_summaries = []

        # Identify which clusters to screen out if screening is requested
        screened_clusters = set()
        if screen_dirs and image_paths:
            prompt_texts = list(prompts.keys())
            screened_clusters = self._identify_screened_clusters(screen_dirs)
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
            representative = self._get_representative_prompt(cluster_prompts)

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
                'cluster_id': int(cluster),
                'size': len(cluster_indices),
                'representative': representative,
                'common_tokens': common[:10] if len(common) > 10 else common,
                'image_path': representative_image,
                'samples': sample_prompts
            })

        return cluster_summaries

    def _identify_screened_clusters(self, screen_dirs):
        """
        Identify clusters that have at least one image in the screen directories.
        """
        normalized_screen_dirs = [os.path.normpath(d) for d in screen_dirs]
        screened_clusters = set()

        # Check each prompt
        for idx, prompt in enumerate(self.prompt_texts):
            cluster_id = self.clusters[idx]

            # Skip noise cluster
            if cluster_id == -1:
                continue

            # Check if this prompt's image is in a screened directory
            image_path = self.image_paths.get(prompt)
            if image_path:
                # Check if this image is in any of the screened directories
                for screen_dir in normalized_screen_dirs:
                    if os.path.normpath(image_path).startswith(screen_dir):
                        screened_clusters.add(cluster_id)
                        break

        return screened_clusters

    def _get_representative_prompt(self, prompts):
        """Select a representative prompt for a cluster at random."""
        return random.choice(prompts) if prompts else ""

    def _analyze_directory_clusters(self, directory_path, clusters, prompt_texts, image_paths):
        """
        Analyze how a specific directory contributes to clusters.
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

    def _analyze_cluster_tag_distribution(self):
        """Analyze tag distribution within each cluster."""
        cluster_tags = {}

        # Group prompts by cluster
        for idx, prompt in enumerate(self.prompt_texts):
            cluster_id = self.clusters[idx]
            if cluster_id not in cluster_tags:
                cluster_tags[cluster_id] = []
            cluster_tags[cluster_id].append(prompt)

        # Extract tags from each cluster's prompts
        cluster_tag_counts = {}
        for cluster_id, prompts in cluster_tags.items():
            cluster_tag_counts[cluster_id] = extract_tags_from_prompts(prompts)

        return cluster_tag_counts

    def _analyze_cluster_pair_diffs(self, max_pairs=10, sample_size=5):
        """
        Analyze tag differences between random pairs of clusters.
        """
        # Get unique clusters (excluding noise)
        unique_clusters = [c for c in np.unique(self.clusters) if c != -1]

        # If we have fewer than 2 clusters, we can't make pairs
        if len(unique_clusters) < 2:
            return {}

        # Choose random pairs
        all_pairs = list(itertools.combinations(unique_clusters, 2))
        random.shuffle(all_pairs)
        selected_pairs = all_pairs[:min(max_pairs, len(all_pairs))]

        # Group prompts by cluster
        cluster_prompts = defaultdict(list)
        for idx, prompt in enumerate(self.prompt_texts):
            cluster_id = self.clusters[idx]
            if cluster_id != -1:  # Skip noise cluster
                cluster_prompts[cluster_id].append(prompt)

        # Analyze differences between pairs
        pair_diffs = {}
        for cluster_a, cluster_b in selected_pairs:
            # Sample prompts from each cluster
            prompts_a = random.sample(cluster_prompts[cluster_a], min(sample_size, len(cluster_prompts[cluster_a])))
            prompts_b = random.sample(cluster_prompts[cluster_b], min(sample_size, len(cluster_prompts[cluster_b])))

            # Extract tags
            tags_a = set()
            for prompt in prompts_a:
                tags_a.update([tag.strip() for tag in prompt.split(',') if tag.strip()])

            tags_b = set()
            for prompt in prompts_b:
                tags_b.update([tag.strip() for tag in prompt.split(',') if tag.strip()])

            # Find differences
            diff_a_not_b = tags_a - tags_b
            diff_b_not_a = tags_b - tags_a

            # Store as a counter for easy analysis
            diff_counter = Counter()
            for tag in diff_a_not_b:
                diff_counter[f"{tag} (in cluster {cluster_a})"] += 1
            for tag in diff_b_not_a:
                diff_counter[f"{tag} (in cluster {cluster_b})"] += 1

            pair_diffs[(cluster_a, cluster_b)] = diff_counter

        return pair_diffs

    def _get_within_cluster_diffs(self, sample_size=10, max_clusters=None):
        """
        Analyze differences within clusters to identify common modifiers.
        """
        # Get unique clusters (excluding noise)
        unique_clusters = [c for c in np.unique(self.clusters) if c != -1]

        if max_clusters is not None:
            random.shuffle(unique_clusters)
            unique_clusters = unique_clusters[:min(max_clusters, len(unique_clusters))]

        # Group prompts by cluster
        cluster_prompts = defaultdict(list)
        for idx, prompt in enumerate(self.prompt_texts):
            cluster_id = self.clusters[idx]
            if cluster_id != -1:  # Skip noise cluster
                cluster_prompts[cluster_id].append(prompt)

        # Collect all diffs
        all_diffs = []

        # For each cluster, sample prompts and compare them
        for cluster_id in unique_clusters:
            cluster_prompt_list = cluster_prompts[cluster_id]

            if len(cluster_prompt_list) < 2:  # Need at least 2 prompts to compare
                continue

            # Sample prompts
            if len(cluster_prompt_list) <= sample_size:
                sampled_prompts = cluster_prompt_list
            else:
                sampled_prompts = random.sample(cluster_prompt_list, sample_size)

            # Generate all pairs from samples
            for i in range(len(sampled_prompts)):
                for j in range(i+1, len(sampled_prompts)):
                    diffs = extract_normalized_diffs(sampled_prompts[i], sampled_prompts[j])
                    all_diffs.extend(diffs)

        return Counter(all_diffs)

    def _get_common_modifiers(self, sample_size=10, max_clusters=None, top_n=50):
        """
        Identify common modifiers across clusters.
        """
        # Get diffs from within clusters
        diffs = self._get_within_cluster_diffs(sample_size, max_clusters)

        # Return top N most common
        return diffs.most_common(top_n)

    def _analyze_modifier_context(self, modifier, max_examples=5):
        """
        Analyze the context in which a modifier is used.
        """
        # Find prompts containing the modifier
        examples = []
        pattern = re.compile(r'\b' + re.escape(modifier) + r'\b')

        for prompt in self.prompt_texts:
            if pattern.search(prompt):
                examples.append(prompt)
                if len(examples) >= max_examples:
                    break

        return examples


def create_analyzer(db_path="data/prompts.sqlite", data_dir="data", force_recompute=False):
    """Create and initialize a TagAnalyzer instance"""
    analyzer = TagAnalyzer(db_path=db_path, data_dir=data_dir)
    analyzer.load_data(force_recompute=force_recompute)
    return analyzer
