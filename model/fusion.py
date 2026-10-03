import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from plan3a.config import EMBED_DIM, SHEAF_HGNN_DIM
from plan3a.data.clinical import get_feature_dim

class ClinicalEncoder(nn.Module):

    def __init__(self, clinical_dim: int = None, embed_dim: int = SHEAF_HGNN_DIM):
        super().__init__()
        if clinical_dim is None:
            clinical_dim = get_feature_dim()

        self.encoder = nn.Sequential(
            nn.Linear(clinical_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.encoder(x)

class DynamicWeighting(nn.Module):

    def __init__(self, embed_dim: int = SHEAF_HGNN_DIM):
        super().__init__()

        self.img_confidence = nn.Linear(embed_dim, 1)
        self.clin_confidence = nn.Linear(embed_dim, 1)

    def forward(
        self,
        img_embed: torch.Tensor,
        clin_embed: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        w_img_mono = torch.sigmoid(self.img_confidence(img_embed))
        w_clin_mono = torch.sigmoid(self.clin_confidence(clin_embed))

        eps = 1e-8

        product = (w_img_mono * w_clin_mono).clamp(min=eps)
        w_img_holo = torch.log(w_img_mono.clamp(min=eps)) / (torch.log(product) + eps)
        w_clin_holo = torch.log(w_clin_mono.clamp(min=eps)) / (torch.log(product) + eps)

        w_img_holo = w_img_holo.clamp(-5, 5)
        w_clin_holo = w_clin_holo.clamp(-5, 5)

        scores = torch.cat([
            w_img_mono + w_img_holo,
            w_clin_mono + w_clin_holo,
        ], dim=-1)
        weights = F.softmax(scores, dim=-1)

        w_img = weights[:, 0:1]
        w_clin = weights[:, 1:2]

        return w_img, w_clin

class InteractiveAlignmentFusion(nn.Module):

    def __init__(self, embed_dim: int = SHEAF_HGNN_DIM, num_heads: int = 4):
        super().__init__()

        self.clin_to_img_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads,
            dropout=0.1, batch_first=True,
        )

        self.img_to_clin_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads,
            dropout=0.1, batch_first=True,
        )

        self.norm_img = nn.LayerNorm(embed_dim)
        self.norm_clin = nn.LayerNorm(embed_dim)

        self.fusion_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

    def forward(
        self,
        img_embed: torch.Tensor,
        clin_embed: torch.Tensor,
    ) -> torch.Tensor:

        img_q = img_embed.unsqueeze(1)
        clin_q = clin_embed.unsqueeze(1)

        img_context, _ = self.clin_to_img_attn(clin_q, img_q, img_q)
        img_fused = self.norm_img(img_embed + img_context.squeeze(1))

        clin_context, _ = self.img_to_clin_attn(img_q, clin_q, clin_q)
        clin_fused = self.norm_clin(clin_embed + clin_context.squeeze(1))

        fused = self.fusion_proj(torch.cat([img_fused, clin_fused], dim=-1))

        return fused

class MultiModalFusion(nn.Module):

    def __init__(
        self,
        embed_dim: int = SHEAF_HGNN_DIM,
        clinical_dim: int = None,
    ):
        super().__init__()
        self.clinical_encoder = ClinicalEncoder(clinical_dim, embed_dim)
        self.dynamic_weighting = DynamicWeighting(embed_dim)
        self.interactive_fusion = InteractiveAlignmentFusion(embed_dim)

    def forward(
        self,
        img_embed: torch.Tensor,
        clinical_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        clin_embed = self.clinical_encoder(clinical_features)

        w_img, w_clin = self.dynamic_weighting(img_embed, clin_embed)

        img_weighted = img_embed * w_img
        clin_weighted = clin_embed * w_clin

        fused = self.interactive_fusion(img_weighted, clin_weighted)

        return {
            "fused": fused,
            "w_img": w_img,
            "w_clin": w_clin,
            "clin_embed": clin_embed,
        }
