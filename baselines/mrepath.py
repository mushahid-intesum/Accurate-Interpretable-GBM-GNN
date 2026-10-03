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

    def __init__(
        self,
        patch_dim: int = 1536,
        embed_dim: int = SHEAF_HGNN_DIM,
        num_layers: int = SHEAF_HGNN_LAYERS,
        clinical_dim: int = None,
        num_survival_bins: int = 4,
    ):
        super().__init__()

        self.shgnn = SheafHGNN(
            in_dim=patch_dim,
            embed_dim=embed_dim,
            num_layers=num_layers,
        )

        self.pooler = GraphPooling(embed_dim)

        self.fusion = MultiModalFusion(embed_dim, clinical_dim)

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

        node_embeds = self.shgnn(
            node_features, hyperedge_index, num_nodes, num_edges,
        )

        graph_embed = self.pooler(node_embeds)

        if clinical_features is not None:
            if clinical_features.dim() == 1:
                clinical_features = clinical_features.unsqueeze(0)
            fusion_out = self.fusion(graph_embed, clinical_features)
            fused = fusion_out["fused"]
            w_img = fusion_out["w_img"]
            w_clin = fusion_out["w_clin"]
        else:
            fused = graph_embed
            w_img = torch.tensor([1.0], device=node_features.device)
            w_clin = torch.tensor([0.0], device=node_features.device)

        hazard_logits = self.survival_head(fused)

        return {

            "hazard_logits": hazard_logits,
            "graph_embed": graph_embed,
            "fused_embed": fused,

            "node_embeddings": node_embeds,

            "w_img": w_img,
            "w_clin": w_clin,

            "concept_loss": torch.tensor(0.0, device=node_features.device),
        }
