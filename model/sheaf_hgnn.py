import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from plan3a.config import (
    EMBED_DIM, SHEAF_HGNN_LAYERS, SHEAF_HGNN_DIM,
    PATCH_ENCODER_CHANNELS, NUM_CONCEPTS,
)

class PatchEncoder(nn.Module):

    def __init__(self, in_dim: int = 1536, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.encoder(x)

class SheafHGNNLayer(nn.Module):

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.vertex_to_edge = nn.Linear(in_dim, out_dim, bias=False)

        self.edge_to_vertex = nn.Linear(out_dim, out_dim, bias=False)

        self.weight = nn.Linear(out_dim, out_dim)

        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(
        self,
        x: torch.Tensor,
        hyperedge_index: torch.Tensor,
        num_nodes: int,
        num_edges: int,
        edge_weights: torch.Tensor = None,
    ) -> torch.Tensor:

        if hyperedge_index.shape[1] == 0:

            return self.activation(self.norm(self.weight(self.vertex_to_edge(x))))

        node_idx = hyperedge_index[0]
        edge_idx = hyperedge_index[1]

        x_transformed = self.vertex_to_edge(x)

        edge_features = torch.zeros(num_edges, self.out_dim, device=x.device)
        edge_counts = torch.zeros(num_edges, 1, device=x.device)

        edge_features.index_add_(0, edge_idx, x_transformed[node_idx])
        edge_counts.index_add_(0, edge_idx, torch.ones(node_idx.shape[0], 1, device=x.device))
        edge_counts = edge_counts.clamp(min=1)
        edge_features = edge_features / edge_counts

        if edge_weights is not None:
            edge_features = edge_features * edge_weights.unsqueeze(-1)

        edge_transformed = self.edge_to_vertex(edge_features)

        node_updates = torch.zeros(num_nodes, self.out_dim, device=x.device)
        node_degree = torch.zeros(num_nodes, 1, device=x.device)

        node_updates.index_add_(0, node_idx, edge_transformed[edge_idx])
        node_degree.index_add_(0, node_idx, torch.ones(node_idx.shape[0], 1, device=x.device))
        node_degree = node_degree.clamp(min=1)

        inv_sqrt_degree = 1.0 / torch.sqrt(node_degree)
        node_updates = node_updates * inv_sqrt_degree

        x_out = self.weight(node_updates)
        x_out = self.norm(x_out)
        x_out = self.activation(x_out)
        x_out = self.dropout(x_out)

        return x_out

class SheafHGNN(nn.Module):

    def __init__(
        self,
        in_dim: int = 1536,
        embed_dim: int = SHEAF_HGNN_DIM,
        num_layers: int = SHEAF_HGNN_LAYERS,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        self.patch_encoder = PatchEncoder(in_dim, embed_dim)

        self.layers = nn.ModuleList([
            SheafHGNNLayer(embed_dim, embed_dim, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * (num_layers + 1), embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        hyperedge_index: torch.Tensor,
        num_nodes: int,
        num_edges: int,
        edge_weights: torch.Tensor = None,
    ) -> torch.Tensor:

        x = self.patch_encoder(node_features)

        layer_outputs = [x]

        for layer in self.layers:
            x_new = layer(x, hyperedge_index, num_nodes, num_edges, edge_weights)

            x = x + x_new
            layer_outputs.append(x)

        x_multi = torch.cat(layer_outputs, dim=-1)
        x = self.fusion(x_multi)

        return x

class GraphPooling(nn.Module):

    def __init__(self, embed_dim: int = SHEAF_HGNN_DIM):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Tanh(),
            nn.Linear(embed_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        attn_scores = self.attention(x)
        attn_weights = F.softmax(attn_scores, dim=0)
        graph_embed = (attn_weights * x).sum(dim=0, keepdim=True)
        return graph_embed
