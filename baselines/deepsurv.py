import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

class DeepSurvModel(nn.Module):

    def __init__(
        self,
        in_dim: int = 18,
        hidden_dims: list = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]

        layers = []
        prev_dim = in_dim

        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.SELU(),
                nn.AlphaDropout(dropout),
                nn.BatchNorm1d(h_dim),
            ])
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.network(x)

class CoxPHLoss(nn.Module):

    def forward(
        self,
        log_risk: torch.Tensor,
        survival_time: torch.Tensor,
        event: torch.Tensor,
    ) -> torch.Tensor:

        log_risk = log_risk.squeeze(-1)

        sorted_idx = torch.argsort(survival_time, descending=True)
        sorted_log_risk = log_risk[sorted_idx]
        sorted_event = event[sorted_idx]

        max_risk = sorted_log_risk.max()
        exp_risk = torch.exp(sorted_log_risk - max_risk)
        cumsum_risk = torch.cumsum(exp_risk, dim=0)
        log_cumsum_risk = torch.log(cumsum_risk + 1e-8) + max_risk

        event_mask = sorted_event == 1
        n_events = event_mask.sum()

        if n_events == 0:
            return torch.tensor(0.0, device=log_risk.device, requires_grad=True)

        partial_ll = sorted_log_risk[event_mask] - log_cumsum_risk[event_mask]
        loss = -partial_ll.mean()

        return loss

class DeepSurvWithNLL(nn.Module):

    def __init__(
        self,
        in_dim: int = 18,
        hidden_dims: list = None,
        num_bins: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]

        layers = []
        prev_dim = in_dim

        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.SELU(),
                nn.AlphaDropout(dropout),
                nn.BatchNorm1d(h_dim),
            ])
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, num_bins))

        self.network = nn.Sequential(*layers)
        self.num_bins = num_bins

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.network(x)
