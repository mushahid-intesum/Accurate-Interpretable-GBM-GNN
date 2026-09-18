"""
CBM Explainer — ante-hoc explanation via concept activation magnitude.

Uses the concept bottleneck's own activations as the explanation:
    importance(node_i) = ||concepts_i||_2

Nodes where the model predicts strong concept values (high enhancement,
high necrosis, etc.) are considered the most important for the prediction.

This is the same logic used in eval/faithfulness.py ExplanationExtractor,
reimplemented here for the unified explainer interface.

For E3 (no EST): loads E3 checkpoint -> concepts without faithfulness training
For E6 (with EST): loads E6 checkpoint -> EST-regularized concepts
Same code, different model weights.
"""
import torch
from typing import Dict

from plan3a.explainability.base import BaseExplainer


class CBMExplainer(BaseExplainer):
    """
    Concept Bottleneck explanation.

    Importance = L2 norm of the 8-dimensional concept activation
    vector at each node. Nodes with high concept magnitude are
    the ones the model considers most informative.
    """

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

        # Importance = L2 norm of concept activations per node
        concept_activations = outputs["concepts"]  # (N, 8)
        importance = torch.norm(concept_activations, dim=-1)  # (N,)

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
