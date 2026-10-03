import torch
import torch.nn.functional as F
from typing import Dict

from explainability.base import BaseExplainer

class AttentionExplainer(BaseExplainer):

    def explain(self, patient_data: Dict) -> Dict[str, torch.Tensor]:
        self.model.eval()
        node_feats, hg, num_nodes, num_edges, concepts, clinical = (
            self._prepare_inputs(patient_data)
        )

        captured = {}

        def _pooling_hook(module, input, output):

            x = input[0]
            attn_scores = module.attention(x)
            captured["pooling_attn_raw"] = attn_scores.detach()
            captured["pooling_attn_weights"] = F.softmax(
                attn_scores, dim=0
            ).detach()

        hook = self.model.pooler.register_forward_hook(_pooling_hook)

        hecrl_hook = None
        if hasattr(self.model, "concept_bottleneck"):
            cb = self.model.concept_bottleneck
            if hasattr(cb, "hecrl") and hasattr(cb.hecrl, "attention"):
                def _hecrl_hook(module, input, output):

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

        if "pooling_attn_weights" in captured:
            importance = captured["pooling_attn_weights"].squeeze(-1)
        else:

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
