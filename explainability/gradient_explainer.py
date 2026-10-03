import torch
from typing import Dict

from explainability.base import BaseExplainer

class GradientExplainer(BaseExplainer):

    def explain(self, patient_data: Dict) -> Dict[str, torch.Tensor]:
        self.model.eval()
        node_feats, hg, num_nodes, num_edges, concepts, clinical = (
            self._prepare_inputs(patient_data)
        )

        node_feats = node_feats.detach().clone().requires_grad_(True)

        outputs = self.model(
            node_features=node_feats,
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

        grad = node_feats.grad
        importance = grad.norm(dim=-1).detach()

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
