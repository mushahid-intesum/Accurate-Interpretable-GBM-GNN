"""
MRePath: Standalone Sheaf Hypergraph + Dynamic Fusion (No Concept Bottleneck).

Standalone baseline for Paper 1 comparison, following the original
MRePath paper (IJCAI 2025).

Pipeline:
    PatchEncoder(1536 -> 64)
    -> SheafHGNN (3 layers, dim=64)     [reused from Plan3a]
    -> AttentionPool (N -> 1)           [reused from Plan3a]
    -> MultiModalFusion (img + clin)    [reused from Plan3a]
    -> SurvivalHead (64 -> 4 bins)      [reused from Plan3a]

Key differences from Plan3a:
  - No concept bottleneck (embeddings go directly to pooling)
  - No concept loss (single-task NLL survival loss only)
  - No EST regularization
  - No ranking loss, no LR warmup (faithful to original paper)
  - Fully opaque: predictions cannot be traced to clinical concepts

This is the "accurate but opaque" baseline. It uses the same graph
structure and fusion as Plan3a but skips the interpretability layer.

Reference: MRePath (IJCAI 2025) — Multimodal Cancer Survival Analysis
via Hypergraph Learning with Cross-Modality Rebalance
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from plan3a.config import SHEAF_HGNN_DIM, SHEAF_HGNN_LAYERS
from plan3a.model.sheaf_hgnn import SheafHGNN, GraphPooling
from plan3a.model.fusion import MultiModalFusion
from plan3a.model.full_model import SurvivalHead, NLLSurvivalLoss


class StandaloneMRePath(nn.Module):
    """
    MRePath: Sheaf Hypergraph + Dynamic Fusion, NO concept bottleneck.

    This is essentially Plan3a E4 without the concept bottleneck.
    The embedding goes directly from SheafHGNN -> pool -> fusion -> survival.

    No concept predictions, no HECRL, no concept loss.
    The model is fully opaque: predictions cannot be traced to clinical concepts.
    """

    def __init__(
        self,
        patch_dim: int = 1536,
        embed_dim: int = SHEAF_HGNN_DIM,
        num_layers: int = SHEAF_HGNN_LAYERS,
        clinical_dim: int = None,
        num_survival_bins: int = 4,
    ):
        super().__init__()

        # Stage 1: SheafHGNN (reused from Plan3a)
        self.shgnn = SheafHGNN(
            in_dim=patch_dim,
            embed_dim=embed_dim,
            num_layers=num_layers,
        )

        # Graph pooling: node embeddings -> graph embedding
        self.pooler = GraphPooling(embed_dim)

        # Stage 2: Multimodal Fusion (reused from Plan3a)
        # Dynamic weighting + interactive alignment fusion
        self.fusion = MultiModalFusion(embed_dim, clinical_dim)

        # Stage 3: Survival Head (reused from Plan3a)
        self.survival_head = SurvivalHead(embed_dim, num_survival_bins)

    def forward(
        self,
        node_features: torch.Tensor,
        hyperedge_index: torch.Tensor,
        num_nodes: int,
        num_edges: int,
        concept_targets: Optional[torch.Tensor] = None,
        clinical_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass.

        Args:
            node_features: (N, 1536) flattened patch features
            hyperedge_index: (2, E) hypergraph incidence
            num_nodes: N
            num_edges: number of hyperedges
            concept_targets: ignored (MRePath has no concept bottleneck)
            clinical_features: (1, clinical_dim) clinical feature vector

        Returns:
            dict with hazard_logits and fusion weights
        """
        # ── Stage 1: SheafHGNN ───────────────────────────────────
        node_embeds = self.shgnn(
            node_features, hyperedge_index, num_nodes, num_edges,
        )  # (N, embed_dim)

        # ── Pool to graph level (skip concept bottleneck) ────────
        graph_embed = self.pooler(node_embeds)  # (1, embed_dim)

        # ── Stage 2: Multimodal Fusion ───────────────────────────
        if clinical_features is not None:
            if clinical_features.dim() == 1:
                clinical_features = clinical_features.unsqueeze(0)
            fusion_out = self.fusion(graph_embed, clinical_features)
            fused = fusion_out["fused"]  # (1, embed_dim)
            w_img = fusion_out["w_img"]
            w_clin = fusion_out["w_clin"]
        else:
            fused = graph_embed
            w_img = torch.tensor([1.0], device=node_features.device)
            w_clin = torch.tensor([0.0], device=node_features.device)

        # ── Stage 3: Survival Head ───────────────────────────────
        hazard_logits = self.survival_head(fused)  # (1, num_bins)

        return {
            # Core outputs
            "hazard_logits": hazard_logits,
            "graph_embed": graph_embed,
            "fused_embed": fused,
            # Node-level
            "node_embeddings": node_embeds,
            # Fusion weights
            "w_img": w_img,
            "w_clin": w_clin,
            # No concept outputs (opaque model)
            "concept_loss": torch.tensor(0.0, device=node_features.device),
        }
