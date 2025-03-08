import os
import numpy as np
import pickle
from pathlib import Path
from typing import Optional

class TagAnalysisData:
    """Class to store and persist tag analysis data."""

    def __init__(self, embeddings: np.ndarray, reduced_embeddings: np.ndarray,
                 clusters: np.ndarray, data_dir: Path):
        """
        Initialize with analysis data.

        Args:
            embeddings: The embeddings for all prompts
            reduced_embeddings: Reduced dimension embeddings for visualization
            clusters: Cluster assignments for all prompts
            data_dir: Directory to save/load data from
        """
        self.embeddings = embeddings
        self.reduced_embeddings = reduced_embeddings
        self.clusters = clusters
        self.data_dir = data_dir

    def _save_analysis_data(self):
        """Save analysis data to disk."""
        # Create directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)

        data_path = os.path.join(self.data_dir, 'analysis_data.pkl')

        data_to_save = {
            'embeddings': self.embeddings,
            'reduced_embeddings': self.reduced_embeddings,
            'clusters': self.clusters
        }

        with open(data_path, 'wb') as f:
            pickle.dump(data_to_save, f)

        print(f"Analysis data saved to {data_path}")

    @classmethod
    def load_analysis_data(cls, data_dir: Path) -> Optional['TagAnalysisData']:
        """
        Load analysis data from disk if available.

        Args:
            data_dir: Directory to load data from

        Returns:
            TagAnalysisData instance if data exists, None otherwise
        """
        data_path = os.path.join(data_dir, 'analysis_data.pkl')

        if not os.path.exists(data_path):
            print(f"No analysis data found at {data_path}")
            return None

        try:
            with open(data_path, 'rb') as f:
                data = pickle.load(f)

            return cls(
                embeddings=data['embeddings'],
                reduced_embeddings=data['reduced_embeddings'],
                clusters=data['clusters'],
                data_dir=data_dir
            )
        except Exception as e:
            print(f"Error loading analysis data: {e}")
            return None


