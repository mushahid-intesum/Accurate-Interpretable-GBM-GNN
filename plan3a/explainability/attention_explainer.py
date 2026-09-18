"""
Attention Explainer — use the model's own attention weights as explanations.

Extracts attention scores from two sources already computed during
the forward pass:

1. GraphPooling.attention: (N, 1) learned attention for graph readout.
   These weights determine how much each node contributes to the
   graph-level embedding. This is what the model literally uses
   to decide which nodes matter for survival prediction.

2. HECRL.attention: concept-level self-attention (optional, for analysis).
   Shows which concept pairs the model considers correlated.

For node-level importance, source 1 (pooling attention) is used.
No gradient computation needed. Just a forward pass with hooks.
"""
import torch
import torch.nn.functional as F
from typing import Dict

from plan3a.explainability.base import BaseExplainer


class AttentionExplainer(BaseExplainer):
    """
    Explanation via the model's internal attention weights.

    Uses a forward hook on GraphPooling to capture the attention
    scores that determine how nodes are weighted during pooling.
    """

    def explain(self, patient_data: Dict) -> Dict[str, torch.Tensor]:
        self.model.eval()
        node_feats, hg, num_nodes, num_edges, concepts, clinical = (
            self._prepare_inputs(patient_data)
        )

        # Storage for captured attention
        captured = {}

        def _pooling_hook(module, input, output):
            """Capture attention scores from GraphPooling."""
            x = input[0]  # (N, embed_dim)
            attn_scores = module.attention(x)  # (N, 1)
            captured["pooling_attn_raw"] = attn_scores.detach()
            captured["pooling_attn_weights"] = F.softmax(
                attn_scores, dim=0
            ).detach()

        # Register hook on the pooler
        hook = self.model.pooler.register_forward_hook(_pooling_hook)

        # Capture HECRL attention if available
        hecrl_hook = None
        if hasattr(self.model, "concept_bottleneck"):
            cb = self.model.concept_bottleneck
            if hasattr(cb, "hecrl") and hasattr(cb.hecrl, "attention"):
                def _hecrl_hook(module, input, output):
                    # MultiheadAttention returns (attn_output, attn_weights)
                    if isinstance(output, tuple) and len(output) >= 2:
                        captured["hecrl_attn"] = output[1].detach()
                hecrl_hook = cb.hecrl.attention.register_forward_hook(_hecrl_hook)

        try:
            with torch.no_grad():
                outputs = self.model(
                    node_features=node_feats,
                    hyperedge_index=hg,
                    num_nodes=num_nodes,
                    num_edges=num_edges,
                    concept_targets=concepts,
                    clinical_features=clinical,
                )
        finally:
            hook.remove()
            if hecrl_hook is not None:
                hecrl_hook.remove()

        # Node importance from pooling attention
        if "pooling_attn_weights" in captured:
            importance = captured["pooling_attn_weights"].squeeze(-1)  # (N,)
        else:
            # Fallback: uniform importance
            importance = torch.ones(num_nodes, device=self.device) / num_nodes

        return {
            "node_importance": importance,
            "explanation_mask": self._to_mask(importance, num_nodes),
            "full_prediction": outputs["hazard_logits"].detach(),
            "metadata": {
                "method": "attention",
                "pooling_attn_raw": captured.get("pooling_attn_raw"),
                "hecrl_attn": captured.get("hecrl_attn"),
                "attn_entropy": -(importance * torch.log(importance + 1e-8)).sum().item(),
            },
        }
