"""
Gradient Explainer — vanilla input gradient explanation.

Importance is the L2 norm of the gradient of the predicted risk
with respect to each node's input features:

    importance(node_i) = ||∂risk/∂x_i||_2

Nodes where small feature perturbations cause large prediction
changes are considered important. This is the simplest possible
post-hoc explanation for differentiable models.
"""
import torch
from typing import Dict

from plan3a.explainability.base import BaseExplainer


class GradientExplainer(BaseExplainer):
    """
    Vanilla input gradient explanation.

    Computes the gradient of the predicted cumulative risk w.r.t.
    each node's input features. Does not require model modification.
    """

    def explain(self, patient_data: Dict) -> Dict[str, torch.Tensor]:
        self.model.eval()
        node_feats, hg, num_nodes, num_edges, concepts, clinical = (
            self._prepare_inputs(patient_data)
        )

        # Enable gradient tracking on input
        node_feats = node_feats.detach().clone().requires_grad_(True)

        # Forward pass (with grad enabled)
        outputs = self.model(
            node_features=node_feats,
            hyperedge_index=hg,
            num_nodes=num_nodes,
            num_edges=num_edges,
            concept_targets=concepts,
            clinical_features=clinical,
        )

        # Compute scalar risk from hazard logits
        hazard = torch.sigmoid(outputs["hazard_logits"])  # (1, K)
        survival = torch.prod(1.0 - hazard, dim=-1)  # (1,)
        risk = 1.0 - survival  # scalar: higher = worse prognosis

        # Backward to get per-node gradients
        risk.backward()

        # Importance = L2 norm of gradient per node
        grad = node_feats.grad  # (N, D)
        importance = grad.norm(dim=-1).detach()  # (N,)

        # Get full prediction (without grad)
        with torch.no_grad():
            full_pred = outputs["hazard_logits"].detach()

        return {
            "node_importance": importance,
            "explanation_mask": self._to_mask(importance, num_nodes),
            "full_prediction": full_pred,
            "metadata": {
                "method": "gradient",
                "grad_norm_mean": importance.mean().item(),
                "grad_norm_max": importance.max().item(),
            },
        }
