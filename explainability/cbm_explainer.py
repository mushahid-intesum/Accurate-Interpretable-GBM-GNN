import torch
from typing import Dict

from explainability.base import BaseExplainer

class CBMExplainer(BaseExplainer):

    def explain(self, patient_data: Dict) -> Dict[str, torch.Tensor]:
        self.model.eval()
        node_feats, hg, num_nodes, num_edges, concepts, clinical = (
            self._prepare_inputs(patient_data)
        )

        with torch.no_grad():
            outputs = self.model(
                node_features=node_feats,
                hyperedge_index=hg,
                num_nodes=num_nodes,
                num_edges=num_edges,
                concept_targets=concepts,
                clinical_features=clinical,
            )

        concept_activations = outputs["concepts"]
        importance = torch.norm(concept_activations, dim=-1)

        return {
            "node_importance": importance,
            "explanation_mask": self._to_mask(importance, num_nodes),
            "full_prediction": outputs["hazard_logits"].detach(),
            "metadata": {
                "method": "cbm",
                "concepts": concept_activations.detach(),
                "concept_raw": outputs.get("concept_raw", concept_activations).detach(),
            },
        }
