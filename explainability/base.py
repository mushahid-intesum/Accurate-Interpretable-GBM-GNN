import torch
from abc import ABC, abstractmethod
from typing import Dict, Optional

class BaseExplainer(ABC):

    def __init__(self, model, top_k_ratio: float = 0.2, device: str = "cpu"):

        self.model = model
        self.top_k_ratio = top_k_ratio
        self.device = device

    @abstractmethod
    def explain(self, patient_data: Dict) -> Dict[str, torch.Tensor]:

        pass

    def _to_mask(
        self, importance: torch.Tensor, num_nodes: int
    ) -> torch.Tensor:

        k = max(1, int(num_nodes * self.top_k_ratio))
        _, top_indices = torch.topk(importance, min(k, len(importance)))
        mask = torch.zeros(num_nodes, dtype=torch.bool, device=importance.device)
        mask[top_indices] = True
        return mask

    def _prepare_inputs(self, patient_data: Dict):

        node_features = patient_data["node_features"].to(self.device)
        hyperedge_index = patient_data["hyperedge_index"].to(self.device)
        num_nodes = patient_data["num_nodes"]
        num_edges = patient_data["num_hyperedges"]
        concepts = patient_data.get("concepts")
        if concepts is not None:
            concepts = concepts.to(self.device)
        clinical = patient_data.get("clinical_features")
        if clinical is not None:
            clinical = clinical.to(self.device)
        return node_features, hyperedge_index, num_nodes, num_edges, concepts, clinical

    def _get_full_prediction(self, patient_data: Dict) -> torch.Tensor:

        self.model.eval()
        node_feats, hg, nn, ne, concepts, clinical = self._prepare_inputs(patient_data)
        with torch.no_grad():
            outputs = self.model(
                node_features=node_feats,
                hyperedge_index=hg,
                num_nodes=nn,
                num_edges=ne,
                concept_targets=concepts,
                clinical_features=clinical,
            )
        return outputs["hazard_logits"].detach()
