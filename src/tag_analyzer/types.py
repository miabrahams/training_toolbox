"""
Data classes for TagAnalyzer.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class SearchResult:
    """Result from a single prompt search match."""
    prompt: str
    image_path: Optional[str]
    cluster: Optional[int]


@dataclass
class SearchResults:
    """Results from searching prompts."""
    query: str
    results: List[SearchResult]
    total_matches: int
    limit_applied: bool
    limit: int


@dataclass
class ClusterStats:
    """Statistics about clusters."""
    total_clusters: int
    displayed_clusters: int
    noise_points: int
    total_prompts: int
    screened_clusters: int = 0
    screen_dirs: Optional[List[str]] = None


@dataclass
class ClusterSummary:
    """Summary of a single cluster."""
    cluster_id: int
    size: int
    representative: str
    common_tokens: List[str]
    image_path: Optional[str]
    samples: List[str]


@dataclass
class ClusterSummaryResults:
    """Results from cluster summary analysis."""
    summaries: List[ClusterSummary]
    stats: ClusterStats


@dataclass
class DirectoryClusterData:
    """Data for a cluster within a directory analysis."""
    count: int
    samples: List[str]


@dataclass
class DirectoryStats:
    """Statistics about directory analysis."""
    total_images: int
    clustered_images: int
    noise_images: int
    cluster_count: int


@dataclass
class DirectoryAnalysisResults:
    """Results from analyzing a directory's contribution to clusters."""
    directory: str
    stats: DirectoryStats
    clusters: Dict[str, DirectoryClusterData]
    noise_samples: List[str]


@dataclass
class ClusterPairDifference:
    """Differences between two clusters."""
    clusters: List[int]
    differences: Dict[str, int]


@dataclass
class TagAnalysisResults:
    """Results from tag analysis."""
    overall_tags: Dict[str, int]
    cluster_tags: Dict[str, Dict[str, int]]
    pair_differences: Dict[str, ClusterPairDifference]


@dataclass
class ModifierExample:
    """Example usage of a modifier."""
    context: str


@dataclass
class ModifierData:
    """Data about a modifier."""
    count: int
    examples: Optional[List[str]] = None


@dataclass
class ModifierAnalysisResults:
    """Results from modifier analysis."""
    modifiers: Dict[str, ModifierData]


@dataclass
class VisualizationPoint:
    """A single point in the visualization."""
    x: float
    y: float
    cluster: int


@dataclass
class DiffInfo:
    """Information about a diff between prompts."""
    prompt_index: int
    diff: str


@dataclass
class ClusterSample:
    """Sample data for a cluster in visualization."""
    size: int
    common_tokens: List[str]
    samples: List[str]
    diffs: Optional[List[DiffInfo]] = None


@dataclass
class VisualizationData:
    """Data for cluster visualization."""
    points: List[VisualizationPoint]
    cluster_samples: Dict[str, ClusterSample]
    total_clusters: int
    total_points: int
    noise_points: int


@dataclass
class ErrorResult:
    """Represents an error result."""
    error: str


def create_error_result(message: str) -> ErrorResult:
    """Helper function to create an error result."""
    return ErrorResult(error=message)