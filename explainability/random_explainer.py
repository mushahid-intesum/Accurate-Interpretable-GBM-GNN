import torch
from typing import Dict

from plan3a.explainability.base import BaseExplainer

class RandomExplainer(BaseExplainer):

    def explain(self, patient_data: Dict) -> Dict[str, torch.Tensor]:
        node_feats, hg, num_nodes, num_edges, concepts, clinical = (
            self._prepare_inputs(patient_data)
        )

        importance = torch.rand(num_nodes, device=self.device)

        full_pred = self._get_full_prediction(patient_data)

        return {
            "node_importance": importance,
            "explanation_mask": self._to_mask(importance, num_nodes),
            "full_prediction": full_pred,
            "metadata": {"method": "random"},
        }
