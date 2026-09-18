"""
Base Explainer — abstract interface for all explanation methods.

Every explainer produces a unified output format:
    node_importance: (N,) float — per-node importance scores
    explanation_mask: (N,) bool — top-k binary mask

This allows any explanation method to be plugged into the
faithfulness audit (Phase 3) without method-specific code.
"""
import torch
from abc import ABC, abstractmethod
from typing import Dict, Optional


class BaseExplainer(ABC):
    """
    Abstract base class for all explanation methods.

    Subclasses must implement `explain()`, which returns per-node
    importance scores and a binary explanation mask.

    The mask selects the top-k fraction of nodes as the "explanation
    subgraph" R ⊆ G. All faithfulness metrics operate on this mask.
    """

    def __init__(self, model, top_k_ratio: float = 0.2, device: str = "cpu"):
        """
        Args:
            model: trained Plan3aModel (or compatible)
            top_k_ratio: fraction of nodes to include in explanation
                         (e.g., 0.2 = top 20% most important nodes)
            device: compute device
        """
        self.model = model
        self.top_k_ratio = top_k_ratio
        self.device = device

    @abstractmethod
    def explain(self, patient_data: Dict) -> Dict[str, torch.Tensor]:
        """
        Generate explanation for one patient.

        Args:
            patient_data: dict from Plan3aDataset with keys:
                node_features, hyperedge_index, num_nodes,
                num_hyperedges, clinical_features, concepts, etc.

        Returns:
            dict with:
                "node_importance": (N,) float — importance per node
                "explanation_mask": (N,) bool — top-k binary mask
                "full_prediction": (K,) hazard logits on full graph
                "metadata": dict — method-specific extras
        """
        pass

    def _to_mask(
        self, importance: torch.Tensor, num_nodes: int
    ) -> torch.Tensor:
        """
        Convert continuous importance scores to a top-k binary mask.

        Args:
            importance: (N,) importance scores (higher = more important)
            num_nodes: N

        Returns:
            mask: (N,) boolean tensor, True for top-k nodes
        """
        k = max(1, int(num_nodes * self.top_k_ratio))
        _, top_indices = torch.topk(importance, min(k, len(importance)))
        mask = torch.zeros(num_nodes, dtype=torch.bool, device=importance.device)
        mask[top_indices] = True
        return mask

    def _prepare_inputs(self, patient_data: Dict):
        """
        Extract and move common inputs to device.

        Returns tuple of:
            (node_features, hyperedge_index, num_nodes, num_edges,
             concepts_target, clinical_features)
        """
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
        """Run frozen model on full graph, return hazard logits."""
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
