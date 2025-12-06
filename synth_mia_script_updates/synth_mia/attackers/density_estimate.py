import numpy as np
import pandas as pd
from scipy import stats
from ..base import BaseAttacker
from typing import Dict, Any, Tuple, List, Optional
from .bnaf.density_estimation import density_estimator_trainer, compute_log_p_x
import torch

class density_estimate(BaseAttacker):
    """
    Black Box Density Estimate Attack.
    Houssiau, F., Jordon, J., Cohen, S. N., Daniel, O., Elliott, A., Geddes, J., Mole, C., Rangel-Smith, C., and Szpruch, L. 
    Tapas: a toolbox for adversarial privacy auditing of synthetic data. 
    arXiv preprint arXiv:2211.06550, 2022.
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
        super().__init__(self.hyper_parameters)    
        self.name = "Density Estimator"

    def _compute_density(self, X_test: np.ndarray, fit_data: np.ndarray) -> np.ndarray:

        method = self.hyper_parameters["estimation_method"]
        device = self.hyper_parameters["device"]

        if method == "kde":
            kde_params = self.hyper_parameters["kde_params"]
            p_fit = stats.gaussian_kde(fit_data.T, **kde_params)
            return p_fit.evaluate(X_test.T)

        elif method == "bnaf":
            bnaf_params = self.hyper_parameters["bnaf_params"]
            _, fit_model = density_estimator_trainer(fit_data, **bnaf_params)
            p_fit_evaluated = np.exp(
                compute_log_p_x(fit_model, torch.as_tensor(X_test).float().to(device))
                .cpu()
                .detach()
                .numpy()
            )
            return p_fit_evaluated

        else:
            raise ValueError(f"Unknown method: {method}. Choose 'bnaf' or 'kde'.")
    

    def _compute_attack_scores(
        self, 
        X_test: np.ndarray, 
        synth: np.ndarray, 
        ref: Optional[np.ndarray] = None
    ) -> np.ndarray:
        return self._compute_density(X_test, synth)
