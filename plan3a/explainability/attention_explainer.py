"""
Attention Explainer — use the model's own attention weights as explanations.

The Plan3a model already computes attention scores in two places:
  1. GraphPooling: (N, 1) attention over nodes for graph-level readout
  2. HECRL: (N, C, C) concept-level self-attention

For node-level importance, we use GraphPooling attention since it directly
determines which nodes the model considers for the final prediction.
This is what the model literally uses to decide which nodes matter.

No additional computation needed, just a forward pass with hooks.
"""
import torch
import torch.nn.functional as F
from typing import Dict

from plan3a.explainability.base import BaseExplainer


class AttentionExplainer(BaseExplainer):
    """
    Explanation via the model's internal attention weights.

    Captures GraphPooling attention scores during a forward pass
    using a forward hook. These scores are the model's own assessment
    of per-node importance for the graph-level representation.
    """

    def explain(self, patient_data: Dict) -> Dict:
        self.model.eval()
        node_features, hg, n_nodes, n_edges, clinical, concepts = \
            self._prepare_inputs(patient_data)

        # Register hook to capture attention scores
        captured_attention = {}

        def _hook_fn(module, input, output):
            # GraphPooling.attention is a Sequential that outputs (N, 1)
            # We want the raw scores before softmax
            # The hook fires on the attention submodule's forward
            captured_attention["raw_scores"] = output.detach()

        # Find the pooler's attention submodule
        pooler = self.model.pooler
        hook = pooler.attention.register_forward_hook(_hook_fn)

        try:
            with torch.no_grad():
                outputs = self.model(
                    node_features=node_features,
                    hyperedge_index=hg,
                    num_nodes=n_nodes,
                    num_edges=n_edges,
                    concept_targets=concepts,
                    clinical_features=clinical,
                )
        finally:
            hook.remove()

        # Extract attention scores
        if "raw_scores" in captured_attention:
            # raw_scores: (N, 1) from GraphPooling.attention
            attn_scores = captured_attention["raw_scores"].squeeze(-1)  # (N,)
            # Normalize to [0, 1] for importance
            attn_weights = F.softmax(attn_scores, dim=0)  # (N,)
            importance = attn_weights
        else:
            # Fallback: uniform importance
            importance = torch.ones(n_nodes, device=self.device) / n_nodes

        mask = self._to_mask(importance, n_nodes)

        return {
            "node_importance": importance,
            "explanation_mask": mask,
            "full_prediction": outputs["hazard_logits"].detach(),
            "metadata": {
                "method": "attention",
                "attn_entropy": float(-(importance * torch.log(importance + 1e-10)).sum()),
            },
        }

    @property
    def name(self):
        return "Attention"
