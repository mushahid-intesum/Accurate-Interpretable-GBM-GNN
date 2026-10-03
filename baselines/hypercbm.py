import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import SHEAF_HGNN_DIM, NUM_CONCEPTS
from model.sheaf_hgnn import PatchEncoder, GraphPooling
from model.concept_bottleneck import ConceptBottleneck
from model.full_model import SurvivalHead, NLLSurvivalLoss

class HGNNPlusLayer(nn.Module):

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.weight = nn.Linear(in_dim, out_dim)

        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(
        self,
        x: torch.Tensor,
        hyperedge_index: torch.Tensor,
        num_nodes: int,
        num_edges: int,
    ) -> torch.Tensor:

        if hyperedge_index.shape[1] == 0:
            return self.activation(self.norm(self.weight(x)))

        node_idx = hyperedge_index[0]
        edge_idx = hyperedge_index[1]

        x_transformed = self.weight(x)

        edge_features = torch.zeros(num_edges, self.out_dim, device=x.device)
        edge_counts = torch.zeros(num_edges, 1, device=x.device)

        edge_features.index_add_(0, edge_idx, x_transformed[node_idx])
        edge_counts.index_add_(
            0, edge_idx,
            torch.ones(node_idx.shape[0], 1, device=x.device),
        )
        edge_counts = edge_counts.clamp(min=1)
        edge_features = edge_features / edge_counts

        node_updates = torch.zeros(num_nodes, self.out_dim, device=x.device)
        node_degree = torch.zeros(num_nodes, 1, device=x.device)

        node_updates.index_add_(0, node_idx, edge_features[edge_idx])
        node_degree.index_add_(
            0, node_idx,
            torch.ones(node_idx.shape[0], 1, device=x.device),
        )
        node_degree = node_degree.clamp(min=1)

        inv_sqrt_degree = 1.0 / torch.sqrt(node_degree)
        node_updates = node_updates * inv_sqrt_degree

        x_out = self.norm(node_updates)
        x_out = self.activation(x_out)
        x_out = self.dropout(x_out)

        return x_out

class HGNNPlus(nn.Module):

    def __init__(
        self,
        in_dim: int = 1536,
        embed_dim: int = SHEAF_HGNN_DIM,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        self.patch_encoder = PatchEncoder(in_dim, embed_dim)

        self.layers = nn.ModuleList([
            HGNNPlusLayer(embed_dim, embed_dim, dropout=dropout)
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
    ) -> torch.Tensor:

        x = self.patch_encoder(node_features)

        layer_outputs = [x]
        for layer in self.layers:
            x_new = layer(x, hyperedge_index, num_nodes, num_edges)
            x = x + x_new
            layer_outputs.append(x)

        x_multi = torch.cat(layer_outputs, dim=-1)
        x = self.fusion(x_multi)

        return x

class StandaloneHyperCBM(nn.Module):

    def __init__(
        self,
        patch_dim: int = 1536,
        embed_dim: int = SHEAF_HGNN_DIM,
        num_layers: int = 3,
        num_concepts: int = NUM_CONCEPTS,
        num_survival_bins: int = 4,
        use_hecrl: bool = True,
    ):
        super().__init__()

        self.hgnn = HGNNPlus(
            in_dim=patch_dim,
            embed_dim=embed_dim,
            num_layers=num_layers,
        )

        self.pooler = GraphPooling(embed_dim)

        self.concept_bottleneck = ConceptBottleneck(
            embed_dim=embed_dim,
            num_concepts=num_concepts,
            use_hecrl=use_hecrl,
        )

        bottleneck_dim = self.concept_bottleneck.output_dim
        self.concept_pool = nn.Sequential(
            nn.Linear(bottleneck_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

        self.survival_head = SurvivalHead(embed_dim, num_survival_bins)

        self.log_var_survival = nn.Parameter(torch.tensor(0.0))
        self.log_var_concept = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        node_features: torch.Tensor,
        hyperedge_index: torch.Tensor,
        num_nodes: int,
        num_edges: int,
        concept_targets: Optional[torch.Tensor] = None,
        clinical_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:

        node_embeds = self.hgnn(
            node_features, hyperedge_index, num_nodes, num_edges,
        )

        concept_out = self.concept_bottleneck(node_embeds, concept_targets)

        concept_graph = self.concept_pool(concept_out["bottleneck_output"])

        graph_embed = self.pooler(concept_graph)

        hazard_logits = self.survival_head(graph_embed)

        return {
            "hazard_logits": hazard_logits,
            "graph_embed": graph_embed,
            "fused_embed": graph_embed,
            "concepts": concept_out["concepts"],
            "concept_raw": concept_out["concept_raw"],
            "concept_loss": concept_out["concept_loss"],
            "node_embeddings": node_embeds,
            "w_img": torch.tensor([1.0], device=node_features.device),
            "w_clin": torch.tensor([0.0], device=node_features.device),
        }

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        survival_time: torch.Tensor,
        event: torch.Tensor,
        time_bins: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        nll_loss = NLLSurvivalLoss(num_bins=outputs["hazard_logits"].shape[-1])
        l_survival = nll_loss(
            outputs["hazard_logits"], survival_time, event, time_bins,
        )

        l_concept = outputs["concept_loss"]

        precision_surv = torch.exp(-self.log_var_survival)
        precision_conc = torch.exp(-self.log_var_concept)

        total_loss = (
            0.5 * precision_surv * l_survival
            + 0.5 * precision_conc * l_concept
            + 0.5 * (self.log_var_survival + self.log_var_concept)
        )

        return {
            "total_loss": total_loss,
            "survival_loss": l_survival,
            "concept_loss": l_concept,
            "log_var_survival": self.log_var_survival,
            "log_var_concept": self.log_var_concept,
        }
