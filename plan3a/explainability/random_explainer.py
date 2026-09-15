"""
Random Explainer — sanity check baseline.

Selects a random subset of nodes as the "explanation".
Should score poorly on all faithfulness metrics (~0.9+ rejection).
If a real method does not beat this, it is not a valid explainer.
"""
import torch
from typing import Dict

from plan3a.explainability.base import BaseExplainer


class RandomExplainer(BaseExplainer):
    """Random node selection baseline."""

    def explain(self, patient_data: Dict) -> Dict:
        num_nodes = patient_data["num_nodes"]
        device = self.device

        # Random importance scores
        importance = torch.rand(num_nodes, device=device)
        mask = self._to_mask(importance, num_nodes)

        # Still need full prediction for faithfulness comparison
        full_pred = self._get_full_prediction(patient_data)

        return {
            "node_importance": importance,
            "explanation_mask": mask,
            "full_prediction": full_pred,
            "metadata": {"method": "random"},
        }

    @property
    def name(self):
        return "Random"
