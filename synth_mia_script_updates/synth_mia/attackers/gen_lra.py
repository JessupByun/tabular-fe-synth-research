import numpy as np
from scipy import stats
from tqdm import tqdm
from ..base import BaseAttacker
from typing import Dict, Any, Tuple, List, Optional
from .bnaf.density_estimation import density_estimator_trainer, compute_log_p_x
import torch 
from sklearn.neighbors import KDTree

class gen_lra(BaseAttacker):
    """
    Shadow-Box Generative Likelihood Ratio Attack (Gen-LRA)
    """
    
    def __init__(self, hyper_parameters=None):
        default_params = {
            'k_nearest': 5,
            "estimation_method": "kde", 
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            "bnaf_params": {
                "epochs": 100,
                "save": False
            },
            "kde_params": {
                "bw_method": "silverman"
            }
        }
        self.hyper_parameters = {**default_params, **(hyper_parameters or {})}
        super().__init__(self.hyper_parameters)    
        self.name = "Gen-LRA"
        
    @staticmethod
    def _find_closest_k_points(X, point, k):
        tree = KDTree(X)
        return tree.query([point], k=k)[1][0]
    
    def _fit_estimator(self, fit_data: np.ndarray):
        method = self.hyper_parameters["estimation_method"]
        
        estimators = {
            "kde": lambda data: stats.gaussian_kde(data.T, **self.hyper_parameters["kde_params"]),
            "bnaf": lambda data: density_estimator_trainer(data, **self.hyper_parameters["bnaf_params"])[1]
        }

        if method not in estimators:
            raise ValueError(f"Unknown method: {method}. Choose 'bnaf' or 'kde'.")

        return estimators[method](fit_data)

    def _compute_density(self, X_test: np.ndarray, model_fit: Any) -> np.ndarray:
        method = self.hyper_parameters["estimation_method"]
        device = self.hyper_parameters["device"]

        density_methods = {
            "kde": lambda X: model_fit.evaluate(X.T),
            "bnaf": lambda X: np.exp(
                compute_log_p_x(model_fit, torch.as_tensor(X).float().to(device)).cpu().detach().numpy()
            )
        }

        if method not in density_methods:
            raise ValueError(f"Unknown method: {method}. Choose 'bnaf' or 'kde'.")

        return density_methods[method](X_test)
    
    def _compute_log_likelihood_ratio(self, likelihoods_H0, likelihoods_Ha):
        epsilon = 1e-20  # Small constant to avoid log(0)

        return np.sum(np.log(likelihoods_Ha+ epsilon)) - np.sum(np.log(likelihoods_H0+ epsilon))
    
    def _compute_attack_scores(
        self, 
        X_test: np.ndarray, 
        synth: np.ndarray, 
        ref: Optional[np.ndarray] = None
    ) -> np.ndarray:
        
        de_H0 = self._fit_estimator(ref)

        results = np.zeros(len(X_test))

        for i, test_point in enumerate(tqdm(X_test, desc="Processing Test dataset")):
            de_Ha = self._fit_estimator(np.vstack([ref, test_point]))
            
            # Get closest k points from synth to x*
            synth_test_indices  = self._find_closest_k_points(synth, test_point, self.hyper_parameters['k_nearest'])
            synth_test = synth[synth_test_indices]

            # Compute likelihoods
            likelihoods_H0 = self._compute_density(synth_test, de_H0)
            likelihoods_Ha = self._compute_density(synth_test, de_Ha)

            # Compute log-likelihood ratio
            results[i] = self._compute_log_likelihood_ratio(likelihoods_H0, likelihoods_Ha)
        
        return results