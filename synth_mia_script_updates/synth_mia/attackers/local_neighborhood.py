import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from ..base import BaseAttacker
from typing import Dict, Any, Tuple, List, Optional

class local_neighborhood(BaseAttacker):
    def __init__(self, hyper_parameters=None):
        # Set default hyperparameters, including the radius r and distance metric
        default_params = {
            'radius': 1.0,  # Default radius
            'metric': 'euclidean'  # Default distance metric
        }
        self.hyper_parameters = {**default_params, **(hyper_parameters or {})}
        super().__init__(self.hyper_parameters)
        self.name = "Local Neighborhood" 
        
    def _compute_attack_scores(self, X_test: np.ndarray, ref: np.ndarray, synth: np.ndarray) -> np.ndarray:
        """
        For each row in X_test, calculate the proportion of points in synth within a sphere of radius r.

        Parameters:
        X_test (np.ndarray): Array of test points.
        synth (np.ndarray): Synthetic dataset.

        Returns:
        np.ndarray: Array of proportions for each test point.
        """
        radius = self.hyper_parameters['radius']
        metric = self.hyper_parameters['metric']

        # Initialize NearestNeighbors with the given metric
        nbrs = NearestNeighbors(radius=radius, metric=metric).fit(synth)

        # Find neighbors within the radius for all points in X_test
        neighbors_within_r = nbrs.radius_neighbors(X_test, return_distance=False)

        # Calculate proportions for all test points in one go
        scores = np.array([len(neighbors) / len(synth) if len(synth) > 0 else 0 for neighbors in neighbors_within_r])

        return scores
