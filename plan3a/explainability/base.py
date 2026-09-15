"""
Base class for all explanation methods.

Every explainer produces a unified output format:
  - node_importance: (N,) float scores, higher = more important
  - explanation_mask: (N,) bool, top-k nodes selected as explanation
  - metadata: dict with method-specific extras

This allows any explainer to be plugged into the faithfulness audit.
"""
import torch
from abc import ABC, abstractmethod
from typing import Dict, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


class BaseExplainer(ABC):
    """
    Abstract base for all explanation methods.

    Subclasses implement `explain()` which returns per-node importance
    scores. The base class provides `_to_mask()` to convert scores
    to a binary top-k mask for faithfulness evaluation.
    """

    def __init__(self, model, top_k_ratio: float = 0.2, device: str = "cpu"):
        """
        Args:
            model: trained Plan3aModel (or compatible)
            top_k_ratio: fraction of nodes to include in explanation
            device: compute device
        """
        self.model = model
        self.top_k_ratio = top_k_ratio
        self.device = device

    @abstractmethod
    def explain(self, patient_data: Dict) -> Dict:
        """
        Generate explanation for one patient.

        Args:
            patient_data: dict from Plan3aDataset with keys:
                node_features, hyperedge_index, num_nodes,
                num_hyperedges, clinical_features, concepts, etc.

        Returns:
            dict with:
                "node_importance": (N,) float importance scores
                "explanation_mask": (N,) bool top-k mask
                "full_prediction": (1, K) hazard logits from full graph
                "metadata": dict with method-specific information
        """
        pass

    def _to_mask(self, importance: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Convert importance scores to binary top-k mask."""
        k = max(1, int(num_nodes * self.top_k_ratio))
        _, top_idx = torch.topk(importance, min(k, len(importance)))
        mask = torch.zeros(num_nodes, dtype=torch.bool, device=importance.device)
        mask[top_idx] = True
        return mask

    def _prepare_inputs(self, patient_data: Dict):
        """Extract and move common inputs to device."""
        node_features = patient_data["node_features"].to(self.device)
        hyperedge_index = patient_data["hyperedge_index"].to(self.device)
        num_nodes = patient_data["num_nodes"]
        num_edges = patient_data["num_hyperedges"]
        clinical = patient_data["clinical_features"].to(self.device)
        concepts = patient_data.get("concepts")
        if concepts is not None:
            concepts = concepts.to(self.device)
        return node_features, hyperedge_index, num_nodes, num_edges, clinical, concepts

    @torch.no_grad()
    def _get_full_prediction(self, patient_data: Dict) -> torch.Tensor:
        """Run model on full graph and return hazard logits."""
        self.model.eval()
        node_features, hg, n_nodes, n_edges, clinical, concepts = \
            self._prepare_inputs(patient_data)

        outputs = self.model(
            node_features=node_features,
            hyperedge_index=hg,
            num_nodes=n_nodes,
            num_edges=n_edges,
            concept_targets=concepts,
            clinical_features=clinical,
        )
        return outputs["hazard_logits"].detach()

    @property
    def name(self) -> str:
        """Human-readable name for this explainer."""
        return self.__class__.__name__
