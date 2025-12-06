import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm
import math
from ..base import BaseAttacker
from typing import Dict, Any, Tuple, List, Optional
from .bnaf.density_estimation import density_estimator_trainer, compute_log_p_x
import torch 

class domias(BaseAttacker):
    """
    Shadow-Box DOMIAS attack. 
    van Breugel, B., Sun, H., Qian, Z., and van der Schaar, M. Membership inference attacks against synthetic data through overfitting detection, 2023. 
    URL https://arxiv.org/abs/2302.12580
    """
    def __init__(self, hyper_parameters=None):
        default_params = {
            "estimation_method": "kde",  # Default to KDE
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
        self.name = "DOMIAS"
        super().__init__(self.hyper_parameters)    

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
    
    def _compute_attack_scores(
        self, 
        X_test: np.ndarray, 
        synth: np.ndarray, 
        ref: Optional[np.ndarray] = None
    ) -> np.ndarray:
        
        de_synth = self._fit_estimator(synth)
        de_ref = self._fit_estimator(ref)

        P_s = self._compute_density(X_test, de_synth)
        P_r = self._compute_density(X_test, de_ref)

        return P_s / (P_r +1e-20)