"""
Gradient Explainer — vanilla input gradient explanation.

importance(node_i) = ||dR/dx_i||_2

where R is the predicted risk score and x_i is node i's input features.
Nodes where small feature perturbations cause large prediction changes
are considered important by the model.

This is the simplest post-hoc gradient-based explanation.
"""
import torch
from typing import Dict

from plan3a.explainability.base import BaseExplainer
from plan3a.eval.task_metrics import hazard_to_risk


class GradientExplainer(BaseExplainer):
    """
    Vanilla gradient explanation.

    Computes the gradient of predicted risk with respect to each node's
    input features, then uses the L2 norm as importance.
    """

    def explain(self, patient_data: Dict) -> Dict:
        self.model.eval()

        # Prepare inputs (need grad on node_features)
        node_features = patient_data["node_features"].to(self.device).clone()
        node_features.requires_grad_(True)

        hg = patient_data["hyperedge_index"].to(self.device)
        n_nodes = patient_data["num_nodes"]
        n_edges = patient_data["num_hyperedges"]
        clinical = patient_data["clinical_features"].to(self.device)
        concepts = patient_data.get("concepts")
        if concepts is not None:
            concepts = concepts.to(self.device)

        # Forward pass (with grad tracking on inputs)
        outputs = self.model(
            node_features=node_features,
            hyperedge_index=hg,
            num_nodes=n_nodes,
            num_edges=n_edges,
            concept_targets=concepts,
            clinical_features=clinical,
        )

        # Compute scalar risk from hazard logits
        hazard_probs = torch.sigmoid(outputs["hazard_logits"])  # (1, K)
        survival = torch.prod(1.0 - hazard_probs, dim=-1)       # (1,)
        risk = 1.0 - survival                                    # (1,) scalar

        # Backward to get gradients w.r.t. node features
        self.model.zero_grad()
        risk.backward()

        # Importance = L2 norm of gradient per node
        grad = node_features.grad  # (N, D)
        importance = grad.norm(dim=-1).detach()  # (N,)

        mask = self._to_mask(importance, n_nodes)

        return {
            "node_importance": importance,
            "explanation_mask": mask,
            "full_prediction": outputs["hazard_logits"].detach(),
            "metadata": {
                "method": "gradient",
                "grad_shape": list(grad.shape),
            },
        }

    @property
    def name(self):
        return "Gradient"
