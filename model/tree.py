import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from plan3a.config import SHEAF_HGNN_DIM, NUM_CONCEPTS

class SoftAssignmentPool(nn.Module):

    def __init__(self, in_dim: int, ratio: float = 0.25):

        super().__init__()
        self.ratio = ratio

        self.assign_net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, in_dim),
        )

        self.feat_transform = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LayerNorm(in_dim),
            nn.GELU(),
        )

    def forward(
        self,
        x: torch.Tensor,
        hyperedge_index: torch.Tensor,
        num_edges: int,
    ) -> Dict[str, torch.Tensor]:

        N, D = x.shape
        K = max(2, int(N * self.ratio))

        assign_logits = self.assign_net(x)

        if D != K:
            proj = nn.Linear(D, K, bias=False).to(x.device)
            nn.init.xavier_uniform_(proj.weight)
            assign_logits = proj(assign_logits)
        S = F.softmax(assign_logits, dim=-1)

        x_coarse = torch.mm(S.t(), x)
        x_coarse = self.feat_transform(x_coarse)

        coarse_he, n_coarse_edges = self._coarsen_hyperedges(
            hyperedge_index, S, K, num_edges
        )

        cluster_assign = S.argmax(dim=-1)

        return {
            "x_coarse": x_coarse,
            "hyperedge_index": coarse_he,
            "num_nodes": K,
            "num_edges": n_coarse_edges,
            "assignment": S,
            "cluster_assign": cluster_assign,
            "assignment_entropy": self._assignment_entropy(S),
        }

    def _coarsen_hyperedges(
        self,
        hyperedge_index: torch.Tensor,
        S: torch.Tensor,
        K: int,
        num_edges: int,
    ) -> Tuple[torch.Tensor, int]:

        if hyperedge_index.shape[1] == 0:
            return torch.zeros(2, 0, dtype=torch.long, device=S.device), 0

        node_idx = hyperedge_index[0]
        edge_idx = hyperedge_index[1]

        cluster_assign = S.argmax(dim=-1)
        coarse_nodes = cluster_assign[node_idx]

        coarse_he = torch.stack([coarse_nodes, edge_idx])

        combined = coarse_he[0] * (num_edges + 1) + coarse_he[1]
        unique_combined, inverse = torch.unique(combined, return_inverse=True)
        coarse_nodes_dedup = unique_combined // (num_edges + 1)
        coarse_edges_dedup = unique_combined % (num_edges + 1)

        unique_edges = torch.unique(coarse_edges_dedup)
        edge_map = torch.zeros(num_edges + 1, dtype=torch.long, device=S.device)
        edge_map[unique_edges] = torch.arange(len(unique_edges), device=S.device)
        coarse_edges_reindexed = edge_map[coarse_edges_dedup]

        return (
            torch.stack([coarse_nodes_dedup, coarse_edges_reindexed]),
            len(unique_edges),
        )

    def _assignment_entropy(self, S: torch.Tensor) -> torch.Tensor:

        eps = 1e-8
        entropy = -(S * torch.log(S + eps)).sum(dim=-1).mean()
        return entropy

class LevelEncoder(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.v2e = nn.Linear(dim, dim, bias=False)
        self.e2v = nn.Linear(dim, dim, bias=False)
        self.weight = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()

    def forward(
        self,
        x: torch.Tensor,
        hyperedge_index: torch.Tensor,
        num_nodes: int,
        num_edges: int,
    ) -> torch.Tensor:

        if hyperedge_index.shape[1] == 0 or num_edges == 0:
            return self.act(self.norm(self.weight(x)))

        node_idx = hyperedge_index[0]
        edge_idx = hyperedge_index[1]
        D = x.shape[-1]

        x_t = self.v2e(x)
        edge_feat = torch.zeros(num_edges, D, device=x.device)
        edge_count = torch.zeros(num_edges, 1, device=x.device)
        edge_feat.index_add_(0, edge_idx, x_t[node_idx])
        edge_count.index_add_(0, edge_idx, torch.ones(node_idx.shape[0], 1, device=x.device))
        edge_feat = edge_feat / edge_count.clamp(min=1)

        edge_out = self.e2v(edge_feat)
        node_upd = torch.zeros(num_nodes, D, device=x.device)
        node_deg = torch.zeros(num_nodes, 1, device=x.device)
        node_upd.index_add_(0, node_idx, edge_out[edge_idx])
        node_deg.index_add_(0, node_idx, torch.ones(node_idx.shape[0], 1, device=x.device))
        node_upd = node_upd / node_deg.clamp(min=1).sqrt()

        return self.act(self.norm(x + self.weight(node_upd)))

class AdaptiveRouter(nn.Module):

    def __init__(self, dim: int, num_levels: int):
        super().__init__()
        self.num_levels = num_levels

        self.level_scorers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim // 2),
                nn.Tanh(),
                nn.Linear(dim // 2, 1),
            )
            for _ in range(num_levels)
        ])

    def forward(self, level_embeds: List[torch.Tensor]) -> Dict[str, torch.Tensor]:

        scores = []
        for i, embed in enumerate(level_embeds):
            score = self.level_scorers[i](embed)
            scores.append(score)

        scores = torch.cat(scores, dim=-1)
        weights = F.softmax(scores, dim=-1)

        stacked = torch.stack(level_embeds, dim=1)
        fused = (weights.unsqueeze(-1) * stacked).sum(dim=1)

        return {
            "fused": fused,
            "routing_weights": weights.squeeze(0),
        }

class MultiGranularTree(nn.Module):

    def __init__(
        self,
        dim: int = SHEAF_HGNN_DIM,
        num_levels: int = 3,
        coarsen_ratio: float = 0.25,
    ):
        super().__init__()
        self.dim = dim
        self.num_levels = num_levels

        self.coarseners = nn.ModuleList([
            SoftAssignmentPool(dim, ratio=coarsen_ratio)
            for _ in range(num_levels)
        ])

        self.level_encoders = nn.ModuleList([
            LevelEncoder(dim) for _ in range(num_levels)
        ])

        self.level_poolers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim // 2),
                nn.Tanh(),
                nn.Linear(dim // 2, 1),
            )
            for _ in range(num_levels + 1)
        ])

        self.router = AdaptiveRouter(dim, num_levels + 1)

    def _attention_pool(self, x: torch.Tensor, pooler: nn.Module) -> torch.Tensor:

        scores = pooler(x)
        weights = F.softmax(scores, dim=0)
        return (weights * x).sum(dim=0, keepdim=True)

    def forward(
        self,
        x: torch.Tensor,
        hyperedge_index: torch.Tensor,
        num_nodes: int,
        num_edges: int,
    ) -> Dict:

        level_embeds = []
        level_data = []

        l0_pool = self._attention_pool(x, self.level_poolers[0])
        level_embeds.append(l0_pool)
        level_data.append({
            "num_nodes": num_nodes,
            "features": x,
        })

        current_x = x
        current_he = hyperedge_index
        current_n = num_nodes
        current_e = num_edges

        for i in range(self.num_levels):

            coarse = self.coarseners[i](current_x, current_he, current_e)

            encoded = self.level_encoders[i](
                coarse["x_coarse"],
                coarse["hyperedge_index"],
                coarse["num_nodes"],
                coarse["num_edges"],
            )

            pool = self._attention_pool(encoded, self.level_poolers[i + 1])
            level_embeds.append(pool)

            level_data.append({
                "num_nodes": coarse["num_nodes"],
                "features": encoded,
                "assignment": coarse["assignment"],
                "cluster_assign": coarse["cluster_assign"],
                "entropy": coarse["assignment_entropy"],
            })

            current_x = encoded
            current_he = coarse["hyperedge_index"]
            current_n = coarse["num_nodes"]
            current_e = coarse["num_edges"]

        routing = self.router(level_embeds)

        return {
            "fused_embed": routing["fused"],
            "routing_weights": routing["routing_weights"],
            "level_embeds": level_embeds,
            "level_data": level_data,
            "num_levels": self.num_levels + 1,
        }
