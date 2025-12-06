import numpy as np
from ..base import BaseAttacker
from typing import Dict, Any, Tuple, List, Optional

class mc(BaseAttacker):
    """
    Black-Box Monte Carlo (MC) Attack
    Hilprecht, B., Harterich, M., and Bernau, D. Monte Carlo and reconstruction membership inference attacks against generative models. 
    Proceedings on Privacy Enhancing Technologies, 2019:232 – 249, 2019. URL https://api.semanticscholar.org/CorpusID:199546273.
    Implementation from: https://arxiv.org/abs/2302.12580

    """
    def __init__(self, hyper_parameters=None):
        # Set default hyperparameters, including the radius r and distance metric
        default_params = {
        }
        self.hyper_parameters = {**default_params, **(hyper_parameters or {})}
        super().__init__(self.hyper_parameters)
        self.name = "MC"
        
    @staticmethod
    def d(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        if len(X.shape) == 1:
            return np.sum((X - Y) ** 2, axis=1)
        else:
            res = np.zeros((X.shape[0], Y.shape[0]))
            for i, x in X:
                res[i] = d(x, Y)
            return res

    def _compute_attack_scores(
        self, 
        X_test: np.ndarray, 
        synth: np.ndarray, 
        ref: Optional[np.ndarray] = None
    ) -> np.ndarray:
        scores = np.zeros(X_test.shape[0])
        distances = np.zeros((X_test.shape[0], synth.shape[0]))
        for i, x in enumerate(X_test):
            distances[i] = mc.d(x, synth)
        # median heuristic (Eq. 4 of Hilprecht)
        min_dist = np.min(distances, 1)
        assert min_dist.size == X_test.shape[0]
        epsilon = np.median(min_dist)
        for i, x in enumerate(X_test):
            scores[i] = np.sum(distances[i] < epsilon)
        scores = scores / synth.shape[0]
        return scores