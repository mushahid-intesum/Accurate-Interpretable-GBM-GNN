"""
HyperCBM: Standalone Hypergraph Concept Bottleneck Model.

Standalone baseline for Paper 1 comparison, following the original
HyperCBM paper architecture.

Key differences from Plan3a:
  - Uses HGNN+ (standard isotropic aggregation), NOT sheaf maps
  - Has concept bottleneck with HECRL (reused from Plan3a)
  - No clinical fusion (concepts only → survival)
  - No EST regularization
  - No ranking loss, no LR warmup
  - NLL survival loss + concept MSE with Kendall weighting only

This isolates the original HyperCBM architecture's performance
when applied to GBM survival prediction.

Reference: HyperCBM (NeurIPS 2026)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from plan3a.config import SHEAF_HGNN_DIM, NUM_CONCEPTS
from plan3a.model.sheaf_hgnn import PatchEncoder, GraphPooling
from plan3a.model.concept_bottleneck import ConceptBottleneck
from plan3a.model.full_model import SurvivalHead, NLLSurvivalLoss


class HGNNPlusLayer(nn.Module):
    """
    Standard HGNN+ layer (isotropic hypergraph convolution).

    Unlike SheafHGNN, this uses isotropic aggregation:
      - No learned sheaf maps F_{v->e} / F_{e->v}
      - Single shared weight matrix for all message passing
      - Simple degree-normalized mean aggregation

    Message passing:
      v -> e: mean(W * x_v for all v in e)
      e -> v: mean(m_e for all e containing v) / sqrt(deg(v))

    This is the standard hypergraph convolution from Feng et al. (2019)
    "Hypergraph Neural Networks" and the HGNN+ variant.
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Single shared transformation (no sheaf maps)
        self.weight = nn.Linear(in_dim, out_dim)

        # Normalization and activation
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
        """
        Args:
            x: (N, in_dim) node features
            hyperedge_index: (2, E_connections) [node_idx; hyperedge_idx]
            num_nodes: N
            num_edges: number of hyperedges

        Returns:
            x_out: (N, out_dim) updated node features
        """
        if hyperedge_index.shape[1] == 0:
            return self.activation(self.norm(self.weight(x)))

        node_idx = hyperedge_index[0]
        edge_idx = hyperedge_index[1]

        # ── Step 1: Transform node features ──────────────────────
        x_transformed = self.weight(x)  # (N, out_dim)

        # ── Step 2: Vertex → Hyperedge (mean aggregation) ────────
        edge_features = torch.zeros(num_edges, self.out_dim, device=x.device)
        edge_counts = torch.zeros(num_edges, 1, device=x.device)

        edge_features.index_add_(0, edge_idx, x_transformed[node_idx])
        edge_counts.index_add_(
            0, edge_idx,
            torch.ones(node_idx.shape[0], 1, device=x.device),
        )
        edge_counts = edge_counts.clamp(min=1)
        edge_features = edge_features / edge_counts  # (E, out_dim)

        # ── Step 3: Hyperedge → Vertex (mean propagation) ────────
        node_updates = torch.zeros(num_nodes, self.out_dim, device=x.device)
        node_degree = torch.zeros(num_nodes, 1, device=x.device)

        node_updates.index_add_(0, node_idx, edge_features[edge_idx])
        node_degree.index_add_(
            0, node_idx,
            torch.ones(node_idx.shape[0], 1, device=x.device),
        )
        node_degree = node_degree.clamp(min=1)

        # ── Step 4: Degree normalization ─────────────────────────
        inv_sqrt_degree = 1.0 / torch.sqrt(node_degree)
        node_updates = node_updates * inv_sqrt_degree

        # ── Step 5: Normalize and activate ───────────────────────
        x_out = self.norm(node_updates)
        x_out = self.activation(x_out)
        x_out = self.dropout(x_out)

        return x_out


class HGNNPlus(nn.Module):
    """
    Multi-layer HGNN+ with PatchEncoder and multi-scale fusion.

    Same structure as SheafHGNN but with HGNNPlusLayer instead
    of SheafHGNNLayer. This removes the sheaf maps entirely.
    """

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

        # Patch encoder: raw patches → embeddings (reused from Plan3a)
        self.patch_encoder = PatchEncoder(in_dim, embed_dim)

        # HGNN+ layers (isotropic, no sheaf maps)
        self.layers = nn.ModuleList([
            HGNNPlusLayer(embed_dim, embed_dim, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Multi-scale fusion: concat all layer outputs → project
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
        """
        Args:
            node_features: (N, 1536) flattened patch features
            hyperedge_index: (2, E_conn) incidence matrix
            num_nodes: N
            num_edges: number of hyperedges

        Returns:
            x: (N, embed_dim) node embeddings
        """
        x = self.patch_encoder(node_features)  # (N, embed_dim)

        layer_outputs = [x]
        for layer in self.layers:
            x_new = layer(x, hyperedge_index, num_nodes, num_edges)
            x = x + x_new  # residual connection
            layer_outputs.append(x)

        # Multi-scale fusion
        x_multi = torch.cat(layer_outputs, dim=-1)
        x = self.fusion(x_multi)

        return x


class StandaloneHyperCBM(nn.Module):
    """
    Standalone HyperCBM following the original paper.

    Pipeline:
        PatchEncoder(1536 -> 64)
        -> HGNN+ (3 layers, dim=64)      [isotropic, no sheaf maps]
        -> ConceptBottleneck + HECRL      [reused from Plan3a]
        -> ConceptPool                    [project concepts to embed space]
        -> AttentionPool                  [N nodes -> 1 graph embedding]
        -> SurvivalHead (64 -> 4 bins)    [reused from Plan3a]

    Key differences from Plan3a E3:
        - HGNN+ instead of SheafHGNN (no sheaf maps)
        - No clinical fusion (concept-only prediction)
        - No EST regularization
        - No ranking loss, no LR warmup
        - Kendall multi-task weighting for survival + concept loss
    """

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

        # Stage 1: HGNN+ (not sheaf)
        self.hgnn = HGNNPlus(
            in_dim=patch_dim,
            embed_dim=embed_dim,
            num_layers=num_layers,
        )

        # Graph pooling
        self.pooler = GraphPooling(embed_dim)

        # Stage 2: Concept Bottleneck (reused from Plan3a)
        self.concept_bottleneck = ConceptBottleneck(
            embed_dim=embed_dim,
            num_concepts=num_concepts,
            use_hecrl=use_hecrl,
        )

        # Concept pooling: pool concept activations to graph level
        bottleneck_dim = self.concept_bottleneck.output_dim
        self.concept_pool = nn.Sequential(
            nn.Linear(bottleneck_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

        # Stage 3: Survival Head (NO fusion, concepts only)
        self.survival_head = SurvivalHead(embed_dim, num_survival_bins)

        # Learned task weights (Kendall)
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
        """
        Full forward pass.

        Args:
            node_features: (N, 1536) flattened patch features
            hyperedge_index: (2, E) hypergraph incidence
            num_nodes: N
            num_edges: number of hyperedges
            concept_targets: (N, 8) precomputed concept GT
            clinical_features: ignored (HyperCBM has no fusion)

        Returns:
            dict with hazard_logits, concepts, and losses
        """
        # ── Stage 1: HGNN+ ──────────────────────────────────────
        node_embeds = self.hgnn(
            node_features, hyperedge_index, num_nodes, num_edges,
        )  # (N, embed_dim)

        # ── Stage 2: Concept Bottleneck ──────────────────────────
        concept_out = self.concept_bottleneck(node_embeds, concept_targets)

        # Pool concepts to graph level
        concept_graph = self.concept_pool(concept_out["bottleneck_output"])

        # Attention-weighted pooling over nodes
        graph_embed = self.pooler(concept_graph)  # (1, embed_dim)

        # ── Stage 3: Survival Head (no fusion) ───────────────────
        hazard_logits = self.survival_head(graph_embed)  # (1, num_bins)

        return {
            "hazard_logits": hazard_logits,
            "graph_embed": graph_embed,
            "fused_embed": graph_embed,  # same as graph_embed (no fusion)
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
        """
        Compute multi-task loss with Kendall weighting.
        Same as Plan3a but without ranking loss or EST.
        """
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
