"""
Random Explainer — sanity check baseline.

Selects a random 20% of nodes as the "explanation".
Should produce high rejection ratios (~0.9+) on all
faithfulness metrics, validating that the audit works.
"""
import torch
from typing import Dict

from plan3a.explainability.base import BaseExplainer


class RandomExplainer(BaseExplainer):
    """
    Explanation by random node selection.

    This is the null hypothesis: if a real explanation method
    does not beat random, it provides no useful signal.
    """

    def explain(self, patient_data: Dict) -> Dict[str, torch.Tensor]:
        node_feats, hg, num_nodes, num_edges, concepts, clinical = (
            self._prepare_inputs(patient_data)
        )

        # Random importance scores
        importance = torch.rand(num_nodes, device=self.device)

        # Get full prediction for reference
        full_pred = self._get_full_prediction(patient_data)

        return {
            "node_importance": importance,
            "explanation_mask": self._to_mask(importance, num_nodes),
            "full_prediction": full_pred,
            "metadata": {"method": "random"},
        }
