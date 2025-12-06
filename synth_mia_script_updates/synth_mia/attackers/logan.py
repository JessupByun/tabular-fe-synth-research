import numpy as np
from ..base import BaseAttacker
import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple, List, Optional

class logan(BaseAttacker):
    """
    Shadow-Box LOGAN attack.
    Hayes, J., Melis, L., Danezis, G., and Cristofaro, E. D. Logan: Membership inference attacks against generative models. 
    Proceedings on Privacy Enhancing Technologies, 2019:133 – 152, 2017. URL https://api.semanticscholar.org/CorpusID:52211986. 
    Implementation from: https://arxiv.org/abs/2302.12580
    """
    def __init__(self, hyper_parameters=None):       
        if hyper_parameters is None:
            hyper_parameters = {}
        super().__init__(hyper_parameters)
        self.name = "LOGAN"
        
    @staticmethod        
    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))
    
    def _compute_attack_scores(
        self, 
        X_test: np.ndarray, 
        synth: np.ndarray, 
        ref: Optional[np.ndarray] = None
    ) -> np.ndarray:
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        num = min(synth.shape[0], ref.shape[0])

        class Net(torch.nn.Module):
            def __init__(
                self, input_dim: int, hidden_dim: int = 256, out_dim: int = 2
            ) -> None:
                super(Net, self).__init__()
                self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
                self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
                self.fc3 = torch.nn.Linear(hidden_dim, out_dim)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                out = self.fc3(x)
                return out

        batch_size = 256
        clf = Net(input_dim=X_test.shape[1]).to(DEVICE)
        optimizer = torch.optim.Adam(clf.parameters(), lr=1e-3)
        loss_func = torch.nn.CrossEntropyLoss()

        all_x, all_y = np.vstack([synth[:num], ref[:num]]), np.concatenate(
            [np.ones(num), np.zeros(num)]
        )
        all_x = torch.as_tensor(all_x).float().to(DEVICE)
        all_y = torch.as_tensor(all_y).long().to(DEVICE)
        X_test = torch.as_tensor(X_test).float().to(DEVICE)
        for training_iter in range(int(300 * len(X_test) / batch_size)):
            rnd_idx = np.random.choice(num, batch_size)
            train_x, train_y = all_x[rnd_idx], all_y[rnd_idx]
            clf_out = clf(train_x)
            loss = loss_func(clf_out, train_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scores = clf(X_test)[:, 1].cpu().detach().numpy()
        torch.cuda.empty_cache()
        return scores

    