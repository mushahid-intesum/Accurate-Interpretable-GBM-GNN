import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from plan3a.config import NUM_CONCEPTS, EMBED_DIM, SHEAF_HGNN_DIM

class ConceptPredictor(nn.Module):

    def __init__(
        self,
        embed_dim: int = SHEAF_HGNN_DIM,
        num_concepts: int = NUM_CONCEPTS,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.num_concepts = num_concepts

        self.shared = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.concept_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 1),
            )
            for _ in range(num_concepts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        shared_feat = self.shared(x)
        concept_preds = []
        for head in self.concept_heads:
            concept_preds.append(head(shared_feat))
        return torch.cat(concept_preds, dim=-1)

class HECRL(nn.Module):

    def __init__(
        self,
        num_concepts: int = NUM_CONCEPTS,
        embed_dim: int = 32,
        num_heads: int = 2,
    ):
        super().__init__()
        self.num_concepts = num_concepts

        self.concept_embed = nn.Linear(1, embed_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.output_proj = nn.Linear(embed_dim, 1)

    def forward(self, concepts: torch.Tensor) -> torch.Tensor:

        N = concepts.shape[0]

        c_tokens = self.concept_embed(concepts.unsqueeze(-1))

        attn_out, _ = self.attention(c_tokens, c_tokens, c_tokens)
        c_refined = self.norm(c_tokens + attn_out)

        refined = self.output_proj(c_refined).squeeze(-1)

        return refined

class ConceptBottleneck(nn.Module):

    def __init__(
        self,
        embed_dim: int = SHEAF_HGNN_DIM,
        num_concepts: int = NUM_CONCEPTS,
        use_hecrl: bool = True,
        residual_bypass: bool = False,
    ):
        super().__init__()
        self.num_concepts = num_concepts
        self.use_hecrl = use_hecrl
        self.residual_bypass = residual_bypass

        self.predictor = ConceptPredictor(embed_dim, num_concepts)

        if use_hecrl:
            self.hecrl = HECRL(num_concepts)

        self.boundary_head = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

        if residual_bypass:
            self.output_dim = num_concepts + embed_dim
        else:
            self.output_dim = num_concepts

    def forward(
        self,
        node_embeddings: torch.Tensor,
        concept_targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:

        concepts_raw = self.predictor(node_embeddings)

        boundary = self.boundary_head(node_embeddings)
        concepts_raw = concepts_raw.clone()
        concepts_raw[:, 6] = boundary.squeeze(-1)

        if self.use_hecrl:
            concepts_refined = self.hecrl(concepts_raw)
        else:
            concepts_refined = concepts_raw

        concept_loss = torch.tensor(0.0, device=node_embeddings.device)
        if concept_targets is not None:

            mask = torch.ones(self.num_concepts, device=node_embeddings.device)
            mask[6] = 0.0

            concept_targets = concept_targets.to(concepts_refined.device)

            diff = (concepts_refined - concept_targets) ** 2
            concept_loss = (diff * mask.unsqueeze(0)).mean()

        if self.residual_bypass:
            bottleneck_output = torch.cat([concepts_refined, node_embeddings], dim=-1)
        else:
            bottleneck_output = concepts_refined

        return {
            "concepts": concepts_refined,
            "concept_loss": concept_loss,
            "bottleneck_output": bottleneck_output,
            "concept_raw": concepts_raw,
        }

class ConceptLoss(nn.Module):

    def __init__(self, num_concepts: int = NUM_CONCEPTS):
        super().__init__()
        self.num_concepts = num_concepts

    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        modality_mask: Optional[Dict[str, bool]] = None,
    ) -> torch.Tensor:

        concept_mask = torch.ones(self.num_concepts, device=predicted.device)
        concept_mask[6] = 0.0

        if modality_mask is not None:
            if not modality_mask.get("DTI", True):
                concept_mask[3] = 0.0
                concept_mask[4] = 0.0
            if not modality_mask.get("Perfusion", True):
                pass

        diff = (predicted - target) ** 2
        masked_diff = diff * concept_mask.unsqueeze(0)

        n_valid = concept_mask.sum().clamp(min=1)
        return masked_diff.sum(dim=-1).mean() / n_valid
