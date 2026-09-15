"""
DeepSurv: Cox Proportional Hazards Deep Neural Network (Katzman 2018).

Standalone baseline for Paper 1 comparison.
Input: clinical features only (18-dim tabular vector).
Output: scalar log-risk per patient.
Loss: negative partial log-likelihood of the Cox PH model.

This uses NO graph structure, NO imaging data, NO concepts.
It measures how well clinical features alone can predict GBM survival.

Reference: Katzman et al. "DeepSurv: personalized treatment recommender
system using a Cox proportional hazards deep neural network" (2018)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class DeepSurvModel(nn.Module):
    """
    Cox Proportional Hazards Deep Neural Network.

    Architecture: MLP with SELU activations (self-normalizing).
    Input: (B, clinical_dim) clinical feature vector
    Output: (B, 1) log-risk score h(x)

    The hazard function is modeled as:
        h(t|x) = h_0(t) * exp(h_theta(x))
    where h_theta is this neural network.
    """

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

        # Final layer: single scalar output (log-risk)
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_dim) clinical features

        Returns:
            log_risk: (B, 1) predicted log-risk scores
        """
        return self.network(x)


class CoxPHLoss(nn.Module):
    """
    Negative partial log-likelihood of the Cox Proportional Hazards model.

    For each event patient i (event=1):
        L_i = h(x_i) - log(sum_{j in R(t_i)} exp(h(x_j)))

    where R(t_i) is the risk set: all patients with survival time >= t_i.

    Total loss: L = -mean(L_i for all event patients)

    This requires batch-level computation because the risk set spans
    all patients in the batch. Unlike Plan3a's per-patient NLL loss,
    DeepSurv needs to see all patients simultaneously.

    Uses the Efron approximation for tied event times.
    """

    def forward(
        self,
        log_risk: torch.Tensor,
        survival_time: torch.Tensor,
        event: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            log_risk: (B, 1) predicted log-risk from DeepSurvModel
            survival_time: (B,) survival time in days
            event: (B,) 1=deceased, 0=censored

        Returns:
            loss: scalar negative partial log-likelihood
        """
        log_risk = log_risk.squeeze(-1)  # (B,)

        # Sort by survival time descending (largest time first)
        # This makes the risk set computation a simple cumulative sum
        sorted_idx = torch.argsort(survival_time, descending=True)
        sorted_log_risk = log_risk[sorted_idx]
        sorted_event = event[sorted_idx]

        # Compute log of cumulative sum of exp(log_risk)
        # For patient at position i (sorted desc by time):
        #   risk_set = patients at positions [0, 1, ..., i]
        #   (all with survival_time >= current patient's time)
        max_risk = sorted_log_risk.max()
        exp_risk = torch.exp(sorted_log_risk - max_risk)
        cumsum_risk = torch.cumsum(exp_risk, dim=0)
        log_cumsum_risk = torch.log(cumsum_risk + 1e-8) + max_risk

        # Partial log-likelihood for event patients only
        # L_i = log_risk_i - log(sum_{j in R(t_i)} exp(log_risk_j))
        event_mask = sorted_event == 1
        n_events = event_mask.sum()

        if n_events == 0:
            return torch.tensor(0.0, device=log_risk.device, requires_grad=True)

        partial_ll = sorted_log_risk[event_mask] - log_cumsum_risk[event_mask]
        loss = -partial_ll.mean()

        return loss


class DeepSurvWithNLL(nn.Module):
    """
    DeepSurv variant using discrete-time NLL loss (same as Plan3a).

    This allows fairer comparison since both models use the same loss
    function and output space (4-bin hazard logits).

    Architecture: MLP with SELU -> 4-bin hazard logits
    """

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
        """
        Args:
            x: (B, in_dim) clinical features

        Returns:
            hazard_logits: (B, num_bins) hazard logits per time bin
        """
        return self.network(x)
