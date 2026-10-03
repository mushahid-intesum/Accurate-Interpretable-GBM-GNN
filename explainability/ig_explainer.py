import torch
from typing import Dict

from explainability.base import BaseExplainer

class IntegratedGradientsExplainer(BaseExplainer):

    def __init__(
        self, model, top_k_ratio: float = 0.2,
        device: str = "cpu", n_steps: int = 50,
    ):

        super().__init__(model, top_k_ratio, device)
        self.n_steps = n_steps

    def explain(self, patient_data: Dict) -> Dict[str, torch.Tensor]:
        self.model.eval()
        node_feats, hg, num_nodes, num_edges, concepts, clinical = (
            self._prepare_inputs(patient_data)
        )

        baseline = torch.zeros_like(node_feats)

        integrated_grads = torch.zeros_like(node_feats)

        for step in range(self.n_steps):

            alpha = step / self.n_steps

            interpolated = baseline + alpha * (node_feats - baseline)
            interpolated = interpolated.detach().clone().requires_grad_(True)

            outputs = self.model(
                node_features=interpolated,
                hyperedge_index=hg,
                num_nodes=num_nodes,
                num_edges=num_edges,
                concept_targets=concepts,
                clinical_features=clinical,
            )

            hazard = torch.sigmoid(outputs["hazard_logits"])
            survival = torch.prod(1.0 - hazard, dim=-1)
            risk = 1.0 - survival

            risk.backward()
            integrated_grads += interpolated.grad.detach()

        ig = (node_feats - baseline) * integrated_grads / self.n_steps

        importance = ig.norm(dim=-1).detach()

        full_pred = self._get_full_prediction(patient_data)

        return {
            "node_importance": importance,
            "explanation_mask": self._to_mask(importance, num_nodes),
            "full_prediction": full_pred,
            "metadata": {
                "method": "integrated_gradients",
                "n_steps": self.n_steps,
                "ig_sum": ig.sum().item(),
                "ig_norm_mean": importance.mean().item(),
                "ig_norm_max": importance.max().item(),
            },
        }
