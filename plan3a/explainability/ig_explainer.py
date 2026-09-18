"""
Integrated Gradients Explainer (Sundararajan et al. 2017).

Computes attributions along a straight-line path from a baseline
(zero features) to the actual input:

    IG(x_i) = (x_i - x'_i) × ∫_{α=0}^{1} ∂F/∂x_i(x' + α(x - x')) dα

where x' is the zero baseline and the integral is approximated
with a Riemann sum over n_steps.

Key property (completeness axiom):
    Σ IG(x_i) = F(x) - F(x')

This means the attributions sum to exactly the prediction difference,
making Integrated Gradients more principled than vanilla gradients.
"""
import torch
from typing import Dict

from plan3a.explainability.base import BaseExplainer


class IntegratedGradientsExplainer(BaseExplainer):
    """
    Integrated Gradients explanation.

    More principled than vanilla gradient: satisfies completeness
    and sensitivity axioms. Uses zero baseline (no features).
    """

    def __init__(
        self, model, top_k_ratio: float = 0.2,
        device: str = "cpu", n_steps: int = 50,
    ):
        """
        Args:
            n_steps: number of interpolation steps for the Riemann sum.
                     Higher = more accurate but slower. 50 is standard.
        """
        super().__init__(model, top_k_ratio, device)
        self.n_steps = n_steps

    def explain(self, patient_data: Dict) -> Dict[str, torch.Tensor]:
        self.model.eval()
        node_feats, hg, num_nodes, num_edges, concepts, clinical = (
            self._prepare_inputs(patient_data)
        )

        # Baseline: zero features (no information)
        baseline = torch.zeros_like(node_feats)

        # Accumulate gradients along the interpolation path
        integrated_grads = torch.zeros_like(node_feats)

        for step in range(self.n_steps):
            # Interpolation coefficient
            alpha = step / self.n_steps

            # Interpolated input along the path
            interpolated = baseline + alpha * (node_feats - baseline)
            interpolated = interpolated.detach().clone().requires_grad_(True)

            # Forward pass
            outputs = self.model(
                node_features=interpolated,
                hyperedge_index=hg,
                num_nodes=num_nodes,
                num_edges=num_edges,
                concept_targets=concepts,
                clinical_features=clinical,
            )

            # Compute scalar risk
            hazard = torch.sigmoid(outputs["hazard_logits"])
            survival = torch.prod(1.0 - hazard, dim=-1)
            risk = 1.0 - survival

            # Backward
            risk.backward()
            integrated_grads += interpolated.grad.detach()

        # Scale by input difference and normalize by n_steps
        # IG = (x - x') * mean(grads along path)
        ig = (node_feats - baseline) * integrated_grads / self.n_steps

        # Node importance = L2 norm of IG per node
        importance = ig.norm(dim=-1).detach()  # (N,)

        # Get full prediction
        full_pred = self._get_full_prediction(patient_data)

        return {
            "node_importance": importance,
            "explanation_mask": self._to_mask(importance, num_nodes),
            "full_prediction": full_pred,
            "metadata": {
                "method": "integrated_gradients",
                "n_steps": self.n_steps,
                "ig_sum": ig.sum().item(),  # should ≈ F(x) - F(baseline)
                "ig_norm_mean": importance.mean().item(),
                "ig_norm_max": importance.max().item(),
            },
        }
