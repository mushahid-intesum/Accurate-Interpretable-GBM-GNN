"""
CBM Explainer — ante-hoc explanation via concept activation magnitude.

Uses the concept bottleneck's own outputs as explanations:
    importance(node_i) = ||concepts_i||_2

Nodes with high concept activation magnitude are considered important
because they contribute the most to the bottleneck's representation.

This is used for both:
  - CBM (E3, no EST): explanation from model trained without EST
  - CBM+EST (E6): explanation from model trained with EST regularization

Same extraction code, different model weights.
"""
import torch
from typing import Dict

from plan3a.explainability.base import BaseExplainer


class CBMExplainer(BaseExplainer):
    """
    Concept Bottleneck explanation.

    Importance = L2 norm of concept activation vector per node.
    This measures how strongly each node activates the clinical concepts.
    """

    def __init__(self, model, top_k_ratio=0.2, device="cpu", label=None):
        """
        Args:
            label: display name override (e.g. "CBM (E3)" or "CBM+EST (E6)")
        """
        super().__init__(model, top_k_ratio, device)
        self._label = label

    def explain(self, patient_data: Dict) -> Dict:
        self.model.eval()
        node_features, hg, n_nodes, n_edges, clinical, concepts = \
            self._prepare_inputs(patient_data)

        with torch.no_grad():
            outputs = self.model(
                node_features=node_features,
                hyperedge_index=hg,
                num_nodes=n_nodes,
                num_edges=n_edges,
                concept_targets=concepts,
                clinical_features=clinical,
            )

        # Importance = L2 norm of concept activations per node
        concept_acts = outputs["concepts"]  # (N, 8)
        importance = torch.norm(concept_acts, dim=-1)  # (N,)
        mask = self._to_mask(importance, n_nodes)

        return {
            "node_importance": importance,
            "explanation_mask": mask,
            "full_prediction": outputs["hazard_logits"].detach(),
            "metadata": {
                "method": "cbm",
                "concepts": concept_acts.detach().cpu(),
                "concept_raw": outputs.get("concept_raw", concept_acts).detach().cpu(),
            },
        }

    @property
    def name(self):
        return self._label or "CBM"
