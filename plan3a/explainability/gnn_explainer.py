"""
Hypergraph GNNExplainer — learned explanation masks for hypergraph GNNs.

Adapted from Ying et al. 2019 "GNNExplainer" for hypergraph structure.

For each patient, learns two sets of soft masks:
1. Hyperedge mask m_e ∈ R^E: which hyperedges matter for the prediction
2. Feature mask m_f ∈ R^D: which input feature dimensions matter

These masks are optimized to preserve the model's original prediction
while being as sparse as possible:

    L = KL(p_orig || p_masked)           # prediction preservation
      + α₁ * ||σ(m_e)||₁                 # hyperedge sparsity
      + α₂ * H(σ(m_e))                   # entropy regularization (push to binary)
      + β  * ||σ(m_f)||₁                 # feature sparsity

Node importance is derived from the hyperedge mask:
    importance(v) = mean(w_e for all e containing v)

Reference: Ying et al. "GNNExplainer: Generating Explanations for
Graph Neural Networks" (NeurIPS 2019), adapted for hypergraphs.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from plan3a.explainability.base import BaseExplainer


class HypergraphGNNExplainer(BaseExplainer):
    """
    Post-hoc explanation learner for trained Plan3a models.

    For each patient, optimizes learnable hyperedge and feature masks
    to find the minimal subgraph that preserves the prediction.
    """

    def __init__(
        self,
        model,
        top_k_ratio: float = 0.2,
        device: str = "cpu",
        lr: float = 0.01,
        optim_epochs: int = 200,
        edge_sparsity: float = 1.0,
        feat_sparsity: float = 0.5,
        entropy_weight: float = 0.1,
    ):
        """
        Args:
            lr: learning rate for mask optimization
            optim_epochs: number of optimization steps per patient
            edge_sparsity: weight for hyperedge mask L1 penalty
            feat_sparsity: weight for feature mask L1 penalty
            entropy_weight: weight for entropy regularization
        """
        super().__init__(model, top_k_ratio, device)
        self.lr = lr
        self.optim_epochs = optim_epochs
        self.edge_sparsity = edge_sparsity
        self.feat_sparsity = feat_sparsity
        self.entropy_weight = entropy_weight

    def explain(self, patient_data: Dict) -> Dict[str, torch.Tensor]:
        self.model.eval()
        # Freeze all model parameters
        for p in self.model.parameters():
            p.requires_grad_(False)

        node_feats, hg, num_nodes, num_edges, concepts, clinical = (
            self._prepare_inputs(patient_data)
        )

        # ── Get original prediction (target to preserve) ─────────
        with torch.no_grad():
            original_outputs = self.model(
                node_features=node_feats,
                hyperedge_index=hg,
                num_nodes=num_nodes,
                num_edges=num_edges,
                concept_targets=concepts,
                clinical_features=clinical,
            )
            original_pred = torch.sigmoid(
                original_outputs["hazard_logits"]
            ).detach()  # (1, K)

        # ── Initialize learnable masks ───────────────────────────
        # Hyperedge mask: initialized to slightly positive (sigmoid ≈ 0.6)
        feat_dim = node_feats.shape[-1]
        mask_edge = nn.Parameter(
            torch.zeros(num_edges, device=self.device) + 0.5
        )
        mask_feat = nn.Parameter(
            torch.zeros(feat_dim, device=self.device) + 0.5
        )

        optimizer = torch.optim.Adam([mask_edge, mask_feat], lr=self.lr)

        # ── Optimization loop ────────────────────────────────────
        best_loss = float("inf")
        best_edge_mask = None
        best_feat_mask = None

        for epoch in range(self.optim_epochs):
            optimizer.zero_grad()

            # Soft masks in [0, 1]
            edge_weights = torch.sigmoid(mask_edge)  # (E,)
            feat_weights = torch.sigmoid(mask_feat)  # (D,)

            # Apply feature mask to input
            masked_features = node_feats * feat_weights.unsqueeze(0)  # (N, D)

            # Forward with edge weights
            masked_outputs = self.model(
                node_features=masked_features,
                hyperedge_index=hg,
                num_nodes=num_nodes,
                num_edges=num_edges,
                concept_targets=concepts,
                clinical_features=clinical,
                edge_weights=edge_weights,
            )
            masked_pred = torch.sigmoid(
                masked_outputs["hazard_logits"]
            )  # (1, K)

            # ── Loss 1: Prediction preservation (KL divergence) ──
            # KL(original || masked) for each bin
            eps = 1e-7
            p = original_pred.clamp(eps, 1 - eps)
            q = masked_pred.clamp(eps, 1 - eps)
            kl_loss = (
                p * torch.log(p / q) + (1 - p) * torch.log((1 - p) / (1 - q))
            ).mean()

            # ── Loss 2: Hyperedge sparsity (L1) ─────────────────
            edge_l1 = edge_weights.mean()

            # ── Loss 3: Entropy regularization (push to binary) ──
            edge_entropy = -(
                edge_weights * torch.log(edge_weights + eps)
                + (1 - edge_weights) * torch.log(1 - edge_weights + eps)
            ).mean()

            # ── Loss 4: Feature sparsity (L1) ───────────────────
            feat_l1 = feat_weights.mean()

            # ── Total loss ───────────────────────────────────────
            loss = (
                kl_loss
                + self.edge_sparsity * edge_l1
                + self.entropy_weight * edge_entropy
                + self.feat_sparsity * feat_l1
            )

            loss.backward()
            optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_edge_mask = edge_weights.detach().clone()
                best_feat_mask = feat_weights.detach().clone()

        # ── Derive node importance from hyperedge mask ───────────
        # For each node, importance = mean weight of its hyperedges
        node_idx = hg[0]
        edge_idx = hg[1]

        node_importance = torch.zeros(num_nodes, device=self.device)
        node_edge_count = torch.zeros(num_nodes, device=self.device)

        # Gather hyperedge weights for each node's connections
        connection_weights = best_edge_mask[edge_idx]  # (E_conn,)
        node_importance.index_add_(0, node_idx, connection_weights)
        node_edge_count.index_add_(
            0, node_idx, torch.ones_like(connection_weights)
        )
        node_edge_count = node_edge_count.clamp(min=1)
        node_importance = node_importance / node_edge_count  # (N,)

        # Re-enable model grad
        for p in self.model.parameters():
            p.requires_grad_(True)

        return {
            "node_importance": node_importance,
            "explanation_mask": self._to_mask(node_importance, num_nodes),
            "full_prediction": original_pred,
            "metadata": {
                "method": "gnn_explainer",
                "edge_mask": best_edge_mask,
                "feat_mask": best_feat_mask,
                "final_loss": best_loss,
                "edge_mask_sparsity": (best_edge_mask < 0.5).float().mean().item(),
                "feat_mask_sparsity": (best_feat_mask < 0.5).float().mean().item(),
                "optim_epochs": self.optim_epochs,
            },
        }
