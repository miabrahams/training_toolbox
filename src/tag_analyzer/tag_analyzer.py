import numpy as np
import random
from collections import Counter, defaultdict
from typing import List, Optional, Callable
import re
import itertools
import os

from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import umap
import hdbscan

from .tag_analysis_data import TagAnalysisData
from .database import TagDatabase
from .utils import (
    extract_tags_from_prompts, common_tokens,
    prompt_diffs, extract_normalized_diffs,
    noCallback
)

from .prompt_data import PromptData

class TagAnalyzer:
    def __init__(self,
                 db_path,
                 data_dir,
                 prompt_data: PromptData,
                 analysis: Optional[TagAnalysisData],
                 db: TagDatabase):
        self.db_path = db_path
        self.prompt_data = prompt_data
        self.data_dir = data_dir
        self.analysis = analysis
        self.db = db

    @property
    def embeddings(self):
        if self.analysis:
            return self.analysis.embeddings
        raise ValueError("Analysis data not loaded.")

    @property
    def reduced_embeddings(self):
        if self.analysis:
            return self.analysis.reduced_embeddings
        raise ValueError("Analysis data not loaded.")

    @property
    def clusters(self):
        if self.analysis:
            return self.analysis.clusters
        raise ValueError("Analysis data not loaded.")

    @property
    def prompt_texts(self):
        return self.prompt_data.prompt_texts

    @property
    def prompts(self):
        return self.prompt_data.prompts

    @property
    def image_paths(self):
        return self.prompt_data.image_paths

    def _compute_analysis_data(self, progress: Callable = noCallback):
        """Process data from scratch and generate embeddings and clusters."""
        progress(0.2, "Loading prompts from database...")

        # Run the analysis
        progress(0.3, "Analyzing prompts...")
        embeddings, reduced_embeddings, clusters = self._analyze_prompts(progress)

        # Create AnalysisData instance
        self.analysis = TagAnalysisData(
            embeddings=embeddings,
            reduced_embeddings=reduced_embeddings,
            clusters=clusters,
            data_dir=self.data_dir
        )

        # Save the analysis data for future use
        progress(0.9, "Saving analysis data...")
        self._save_analysis_data()

        return True

    def search_prompts(self, query, case_sensitive=False, limit=500, progress: Callable = noCallback):
        """
        Search through prompts for a given query string.

        Args:
            query: The search string to look for
            case_sensitive: Whether to perform case-sensitive search
            limit: Maximum number of results to return
            progress: Progress callback function

        Returns:
            Dict containing search results and stats
        """
        if not query:
            return {"error": "Empty search query"}

        progress(0.2, f"Searching prompts for: {query}")

        # Prepare for search
        results = []
        match_count = 0

        # Convert query to lowercase if case-insensitive search
        if not case_sensitive:
            search_query = query.lower()
        else:
            search_query = query

        # Search through prompts
        progress(0.4, "Processing prompts...")
        for idx, prompt in enumerate(self.prompt_data.prompt_texts):
            # Apply case sensitivity setting
            compare_prompt = prompt if case_sensitive else prompt.lower()

            if search_query in compare_prompt:
                match_count += 1

                # Add to results if under limit
                if len(results) < limit:
                    image_path = self.prompt_data.image_paths.get(prompt, None)

                    # Get cluster information if available - only if analysis is loaded
                    cluster_id = None
                    if self.analysis and idx < len(self.analysis.clusters):
                        cluster_id = int(self.analysis.clusters[idx])

                    results.append({
                        "prompt": prompt,
                        "image_path": image_path,
                        "cluster": cluster_id
                    })

        progress(1.0, f"Found {match_count} matches")

        return {
            "query": query,
            "results": results,
            "total_matches": match_count,
            "limit_applied": match_count > limit,
            "limit": limit
        }

    def _analyze_prompts(self, progress: Callable = noCallback):
        """Generate embeddings, reduce dimensions, and cluster the prompts."""
        # Generate embeddings
        progress(0.4, "Generating embeddings...")

        embeddings = self._generate_embeddings(self.prompt_texts)
        progress(0.6, "Reducing dimensions...")

        # Reduce dimensions for visualization
        reduced_embeddings = self._reduce_dimensions(embeddings)
        progress(0.8, "Generating clusters...")

        # Cluster the embeddings
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

    def _save_analysis_data(self):
        """Save analysis data to disk using the AnalysisData instance."""
        if self.analysis is not None:
            self.analysis._save_analysis_data()

    def get_cluster_summary(self, sample_size=5, screen_dirs=None, show_paths=False, progress: Callable = noCallback):
        """Generate cluster summaries"""
        if self.analysis is None:
            return {"error": "Data not loaded. Call load_data() first."}

        progress(0.3, "Generating cluster summaries...")

        cluster_summaries = self._make_cluster_summaries(
            self.clusters, self.prompts, sample_size,
            image_paths=self.image_paths,
            screen_dirs=screen_dirs
        )

        # Add statistics
        n_clusters = len(np.unique(self.clusters)) - (1 if -1 in self.clusters else 0)
        noise_points = np.sum(self.clusters == -1)
        total_displayed = len(cluster_summaries)

        progress(1.0, "Cluster summary generated!")

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

    def analyze_directory(self, directory, sample_size=5, noise_sample=10, progress: Callable = noCallback):
        """Analyze directory contributions to clusters"""
        if not self.analysis is not None:
            return {"error": "Data not loaded. Call load_data() first."}

        progress(0.3, f"Analyzing directory: {directory}...")

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

        progress(1.0, "Directory analysis complete!")

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

    def analyze_tags(self, top_n=20, include_noise=False, cluster_pairs=5, sample_size=10, progress: Callable = noCallback):
        """Analyze tag distribution"""
        if self.analysis is not None:
            return {"error": "Data not loaded. Call load_data() first."}

        progress(0.3, "Analyzing tag distribution...")

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

        progress(1.0, "Tag analysis complete!")

        return {
            "overall_tags": {tag: count for tag, count in most_common},
            "cluster_tags": cluster_tag_data,
            "pair_differences": pair_diffs
        }

    def analyze_modifiers(self, top_n=50, sample_size=20, max_clusters=None,
                          show_examples=False, max_examples=3, progress: Callable = noCallback):
        """Analyze common modifiers across clusters"""
        if self.analysis is not None:
            return {"error": "Data not loaded. Call load_data() first."}

        progress(0.3, "Analyzing tag modifiers...")

        modifiers = self._get_common_modifiers(
            sample_size=sample_size,
            max_clusters=max_clusters if max_clusters and max_clusters > 0 else None,
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

        progress(1.0, "Modifier analysis complete!")

        return result

    def generate_visualization(self, sample_size=100, directory=None, with_diffs=False, progress_callback: Callable = noCallback):
        """Generate visualization data"""
        if self.analysis is not None:
            return {"error": "Data not loaded. Call load_data() first."}

        progress_callback(0.3, "Generating visualization...")

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
                result = self._prepare_visualization_data(
                    filtered_reduced, filtered_clusters, filtered_prompts, sample_size, with_diffs
                )
            else:
                result = {"error": f"No prompts found in directory: {directory}"}
        else:
            # Use all data
            result = self._prepare_visualization_data(
                self.reduced_embeddings, self.clusters, self.prompt_texts, sample_size, with_diffs
            )

        progress_callback(1.0, "Visualization complete!")

        return result

    def generate_plot(self, sample_size=100, directory=None, with_diffs=False, progress_callback: Callable = noCallback):
        """Generate cluster visualization for UI"""
        if self.analysis is not None:
            return None, "Error: Analyzer not initialized. Please initialize first."

        try:
            import matplotlib.pyplot as plt
            from io import BytesIO
            import PIL.Image as Image

            # Generate visualization data
            vis_data = self.generate_visualization(
                sample_size=sample_size,
                directory=directory,
                with_diffs=with_diffs,
                progress_callback=progress_callback
            )

            if "error" in vis_data:
                return None, vis_data["error"]

            # Create figure
            fig = plt.figure(figsize=(10, 8))

            # Extract point data
            x = [p["x"] for p in vis_data["points"]]
            y = [p["y"] for p in vis_data["points"]]
            clusters = [p["cluster"] for p in vis_data["points"]]

            # Plot scatter
            scatter = plt.scatter(x, y, c=clusters, cmap='tab20', alpha=0.6, s=10)
            plt.colorbar(scatter, label="Cluster")
            plt.title('Prompt Clusters')
            plt.xlabel('UMAP Dimension 1')
            plt.ylabel('UMAP Dimension 2')

            # Create text output
            text_output = f"Displaying {len(vis_data['points'])} points in {vis_data['total_clusters']} clusters"
            if vis_data["noise_points"] > 0:
                text_output += f"\nNoise points: {vis_data['noise_points']}"

            if directory:
                text_output += f"\nFiltered by directory: {directory}"

            # Convert plot to image
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            img = Image.open(buf)
            plt.close(fig)

            return img, text_output

        except Exception as e:
            return None, f"Error generating plot: {str(e)}"

    def _prepare_visualization_data(self, reduced_embeddings, clusters, prompt_texts, sample_size=100, with_diffs=False):
        """Prepare data for visualization"""
        # Limit the number of points to visualize
        max_points = min(5000, len(reduced_embeddings))

        if len(reduced_embeddings) > max_points:
            # Take a random sample of points to avoid overwhelming the UI
            indices = np.random.choice(len(reduced_embeddings), max_points, replace=False)
            reduced_embeddings = reduced_embeddings[indices]
            clusters = clusters[indices]
            prompt_texts = [prompt_texts[i] for i in indices]

        # Convert the embeddings to a list format that can be easily sent to JS
        points = []
        for i in range(len(reduced_embeddings)):
            points.append({
                "x": float(reduced_embeddings[i, 0]),
                "y": float(reduced_embeddings[i, 1]),
                "cluster": int(clusters[i]),
                # Don't include full prompt text to reduce data size
                # "prompt": prompt_texts[i] if i < len(prompt_texts) else ""
            })

        # Get cluster samples
        cluster_samples = {}
        unique_clusters = np.unique(clusters)

        for cluster_id in unique_clusters:
            if cluster_id == -1:
                continue  # Skip noise cluster

            # Get indices for this cluster
            cluster_indices = np.where(clusters == cluster_id)[0]

            # Limit samples for large clusters
            sample_size_for_cluster = min(sample_size, len(cluster_indices), 20)  # Hard cap at 20 samples

            # Sample prompts
            sample_indices = random.sample(
                list(cluster_indices),
                sample_size_for_cluster
            )

            sample_prompts = [prompt_texts[i] for i in sample_indices]

            # Get common tokens - limit to 10 maximum
            common = common_tokens(sample_prompts)[:10]

            # Store samples - limit to 5 samples maximum
            cluster_samples[str(cluster_id)] = {
                "size": len(cluster_indices),
                "common_tokens": common,
                "samples": sample_prompts[:5]  # Limit to 5 samples
            }

            # Add diff analysis if requested - limit to 3 maximum
            if with_diffs and sample_prompts:
                diffs = []
                baseline = sample_prompts[0]
                for i, prompt in enumerate(sample_prompts[1:3]):  # Limit to 2 diffs only
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


    def _make_cluster_summaries(self, clusters, prompts, sample_size=5,
                               image_paths=None, screen_dirs=None):
        """Generate summaries for each cluster with representative prompts."""
        unique_clusters = np.unique(clusters)
        cluster_summaries = []

        # Identify which clusters to screen out if screening is requested
        screened_clusters = set()
        if screen_dirs and image_paths:
            screened_clusters = self._identify_screened_clusters(screen_dirs)
            print(f"Found {len(screened_clusters)} clusters represented in screened directories")

        for cluster in unique_clusters:
            if cluster == -1:
                continue  # Skip noise cluster

            # Skip this cluster if it's in the screened set
            if screen_dirs and cluster in screened_clusters:
                continue

            cluster_indices = np.where(clusters == cluster)[0]
            cluster_prompts = [self.prompt_texts[i] for i in cluster_indices]

            # Select a representative prompt
            representative = self._get_representative_prompt(cluster_prompts)

            # Get sample prompts for analysis
            sample_indices = random.sample(list(cluster_indices), min(sample_size, len(cluster_indices)))
            sample_prompts = [self.prompt_texts[i] for i in sample_indices]

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


def create_analyzer(
    db_path,
    data_dir,
    prompt_data,
    db,
    compute_analysis: bool = False,
    progress: Callable = noCallback,
):
    """
    Create and initialize a TagAnalyzer instance

    Args:
        db_path: Path to the database file
        data_dir: Path to directory to store/load analysis data
        prompt_data: PromptData instance with prompt texts and paths
        db: TagDatabase instance
        compute_analysis: Whether to compute embeddings and clusters immediately
        progress: Progress callback function

    Returns:
        Initialized TagAnalyzer instance
    """
    # Try to load existing analysis data
    analysis_data = TagAnalysisData.load_analysis_data(data_dir)

    analyzer = TagAnalyzer(
        db_path=db_path,
        data_dir=data_dir,
        prompt_data=prompt_data,
        analysis=analysis_data,
        db=db
    )

    # Only compute analysis data if requested and not already loaded
    if compute_analysis and analysis_data is None:
        analyzer._compute_analysis_data(progress)

    return analyzer