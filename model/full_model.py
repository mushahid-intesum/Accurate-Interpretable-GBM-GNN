import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from plan3a.config import SHEAF_HGNN_DIM, SHEAF_HGNN_LAYERS, NUM_CONCEPTS
from plan3a.model.sheaf_hgnn import SheafHGNN, GraphPooling
from plan3a.model.concept_bottleneck import ConceptBottleneck
from plan3a.model.fusion import MultiModalFusion
from plan3a.model.tree import MultiGranularTree

class SurvivalHead(nn.Module):

    def __init__(self, in_dim: int, num_bins: int = 4, dropout: float = 0.2):
        super().__init__()
        self.num_bins = num_bins
        self.head = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_bins),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.head(x)

class NLLSurvivalLoss(nn.Module):

    def __init__(self, num_bins: int = 4):
        super().__init__()
        self.num_bins = num_bins

    def forward(
        self,
        hazard_logits: torch.Tensor,
        survival_time: torch.Tensor,
        event: torch.Tensor,
        time_bins: torch.Tensor,
    ) -> torch.Tensor:

        hazard = torch.sigmoid(hazard_logits)
        hazard = hazard.clamp(1e-7, 1 - 1e-7)

        B = hazard.shape[0]

        bin_idx = torch.bucketize(survival_time, time_bins) - 1
        bin_idx = bin_idx.clamp(0, self.num_bins - 1)

        loss = torch.tensor(0.0, device=hazard.device)

        for i in range(B):
            k = bin_idx[i].item()
            is_event = event[i].item()

            survival_part = torch.tensor(0.0, device=hazard.device)
            for j in range(k):
                survival_part += torch.log(1 - hazard[i, j])

            if is_event == 1:

                loss += -torch.log(hazard[i, k]) - survival_part
            else:

                loss += -survival_part - torch.log(1 - hazard[i, k])

        return loss / max(B, 1)

class CoxRankingLoss(nn.Module):

    def __init__(self, margin: float = 0.0):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        risk_scores: torch.Tensor,
        survival_times: torch.Tensor,
        events: torch.Tensor,
    ) -> torch.Tensor:

        N = risk_scores.shape[0]
        if N < 2:
            return torch.tensor(0.0, device=risk_scores.device, requires_grad=True)

        loss = torch.tensor(0.0, device=risk_scores.device)
        n_pairs = 0

        for i in range(N):
            if events[i].item() != 1:
                continue

            mask = survival_times > survival_times[i]
            if not mask.any():
                continue

            diff = risk_scores[i] - risk_scores[mask] - self.margin
            pair_loss = -F.logsigmoid(diff)
            loss = loss + pair_loss.sum()
            n_pairs += mask.sum().item()

        if n_pairs == 0:
            return torch.tensor(0.0, device=risk_scores.device, requires_grad=True)

        return loss / n_pairs

class Plan3aModel(nn.Module):

    def __init__(
        self,
        patch_dim: int = 1536,
        embed_dim: int = SHEAF_HGNN_DIM,
        num_layers: int = SHEAF_HGNN_LAYERS,
        num_concepts: int = NUM_CONCEPTS,
        clinical_dim: int = None,
        num_survival_bins: int = 4,
        use_hecrl: bool = True,
        residual_bypass: bool = False,
        use_fusion: bool = True,
        use_tree: bool = False,
        tree_levels: int = 3,
    ):
        super().__init__()
        self.use_fusion = use_fusion
        self.use_tree = use_tree

        self.shgnn = SheafHGNN(
            in_dim=patch_dim,
            embed_dim=embed_dim,
            num_layers=num_layers,
        )

        self.pooler = GraphPooling(embed_dim)

        self.concept_bottleneck = ConceptBottleneck(
            embed_dim=embed_dim,
            num_concepts=num_concepts,
            use_hecrl=use_hecrl,
            residual_bypass=residual_bypass,
        )

        bottleneck_dim = self.concept_bottleneck.output_dim
        self.concept_pool = nn.Sequential(
            nn.Linear(bottleneck_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

        if use_tree:
            self.tree = MultiGranularTree(
                dim=embed_dim, num_levels=tree_levels, coarsen_ratio=0.25,
            )
        else:
            self.tree = None

        if use_fusion:
            self.fusion = MultiModalFusion(embed_dim, clinical_dim)
            survival_in_dim = embed_dim
        else:
            self.fusion = None
            survival_in_dim = embed_dim

        self.survival_head = SurvivalHead(survival_in_dim, num_survival_bins)

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
        edge_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:

        node_embeds = self.shgnn(
            node_features, hyperedge_index, num_nodes, num_edges, edge_weights
        )

        concept_out = self.concept_bottleneck(node_embeds, concept_targets)

        concept_graph = self.concept_pool(concept_out["bottleneck_output"])

        tree_output = None
        if self.use_tree and self.tree is not None:
            tree_output = self.tree(
                concept_graph, hyperedge_index, num_nodes, num_edges
            )
            graph_embed = tree_output["fused_embed"]
        else:

            graph_embed = self.pooler(concept_graph)

        if self.use_fusion and clinical_features is not None:
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

        result = {

            "hazard_logits": hazard_logits,
            "graph_embed": graph_embed,
            "fused_embed": fused,

            "concepts": concept_out["concepts"],
            "concept_raw": concept_out["concept_raw"],
            "concept_loss": concept_out["concept_loss"],

            "node_embeddings": node_embeds,

            "w_img": w_img,
            "w_clin": w_clin,
        }

        if tree_output is not None:
            result["routing_weights"] = tree_output["routing_weights"]
            result["tree_level_data"] = tree_output["level_data"]
            result["num_tree_levels"] = tree_output["num_levels"]

        return result

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        survival_time: torch.Tensor,
        event: torch.Tensor,
        time_bins: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        nll_loss = NLLSurvivalLoss(num_bins=outputs["hazard_logits"].shape[-1])
        l_survival = nll_loss(
            outputs["hazard_logits"], survival_time, event, time_bins
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
