import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from plan3a.config import SHEAF_HGNN_DIM

class ExplanationExtractor:

    def __init__(self, top_k_ratio: float = 0.2):

        self.top_k_ratio = top_k_ratio

    @torch.no_grad()
    def extract(
        self,
        model,
        node_features: torch.Tensor,
        hyperedge_index: torch.Tensor,
        num_nodes: int,
        num_edges: int,
        concepts_target: torch.Tensor = None,
        clinical_features: torch.Tensor = None,
    ) -> Dict:

        model.eval()

        outputs = model(
            node_features=node_features,
            hyperedge_index=hyperedge_index,
            num_nodes=num_nodes,
            num_edges=num_edges,
            concept_targets=concepts_target,
            clinical_features=clinical_features,
        )

        concepts = outputs["concepts"]
        importance = torch.norm(concepts, dim=-1)

        k = max(1, int(num_nodes * self.top_k_ratio))
        _, top_indices = torch.topk(importance, k)
        explanation_mask = torch.zeros(num_nodes, dtype=torch.bool,
                                       device=node_features.device)
        explanation_mask[top_indices] = True

        return {
            "explanation_mask": explanation_mask,
            "importance_scores": importance,
            "full_output": outputs,
            "full_prediction": outputs["hazard_logits"].detach(),
        }

def _filter_hyperedges(
    hyperedge_index: torch.Tensor,
    node_mask: torch.Tensor,
    num_edges: int,
) -> Tuple[torch.Tensor, int]:

    if hyperedge_index.shape[1] == 0:
        return hyperedge_index, 0

    node_idx = hyperedge_index[0]
    edge_idx = hyperedge_index[1]

    keep = node_mask[node_idx]
    new_node_idx = node_idx[keep]
    new_edge_idx = edge_idx[keep]

    if new_node_idx.shape[0] == 0:
        return torch.zeros(2, 0, dtype=torch.long, device=hyperedge_index.device), 0

    unique_edges = torch.unique(new_edge_idx)
    edge_mapping = torch.zeros(num_edges, dtype=torch.long, device=hyperedge_index.device)
    edge_mapping[unique_edges] = torch.arange(len(unique_edges), device=hyperedge_index.device)
    new_edge_idx = edge_mapping[new_edge_idx]

    return torch.stack([new_node_idx, new_edge_idx]), len(unique_edges)

class FaithfulnessAuditor:

    def __init__(
        self,
        model,
        device: str = "cpu",
        est_samples: int = 50,
        rfid_p: float = 0.9,
        prediction_threshold: float = 0.1,
    ):

        self.model = model
        self.device = device
        self.est_samples = est_samples
        self.rfid_p = rfid_p
        self.threshold = prediction_threshold

    @torch.no_grad()
    def _get_prediction(
        self,
        node_features: torch.Tensor,
        hyperedge_index: torch.Tensor,
        num_nodes: int,
        num_edges: int,
        clinical_features: torch.Tensor = None,
    ) -> torch.Tensor:

        self.model.eval()
        outputs = self.model(
            node_features=node_features,
            hyperedge_index=hyperedge_index,
            num_nodes=num_nodes,
            num_edges=num_edges,
            clinical_features=clinical_features,
        )
        return outputs["hazard_logits"].detach()

    def _prediction_shift(
        self,
        pred_a: torch.Tensor,
        pred_b: torch.Tensor,
    ) -> float:

        p_a = torch.sigmoid(pred_a)
        p_b = torch.sigmoid(pred_b)
        return float((p_a - p_b).abs().mean())

    @staticmethod
    def _extract_subgraph(
        node_features: torch.Tensor,
        hyperedge_index: torch.Tensor,
        node_mask: torch.Tensor,
        num_edges: int,
    ):

        selected_indices = torch.where(node_mask)[0]
        sub_features = node_features[selected_indices]
        K = len(selected_indices)

        if hyperedge_index.shape[1] == 0 or K == 0:
            empty_he = torch.zeros(2, 0, dtype=torch.long,
                                   device=node_features.device)
            return sub_features, empty_he, K, 0

        node_remap = torch.full((node_features.shape[0],), -1,
                                dtype=torch.long, device=node_features.device)
        node_remap[selected_indices] = torch.arange(K, device=node_features.device)

        old_node_idx = hyperedge_index[0]
        old_edge_idx = hyperedge_index[1]
        keep = node_mask[old_node_idx]

        new_node_idx = node_remap[old_node_idx[keep]]
        new_edge_idx = old_edge_idx[keep]

        if new_node_idx.shape[0] == 0:
            empty_he = torch.zeros(2, 0, dtype=torch.long,
                                   device=node_features.device)
            return sub_features, empty_he, K, 0

        unique_edges = torch.unique(new_edge_idx)
        edge_remap = torch.zeros(num_edges, dtype=torch.long,
                                 device=node_features.device)
        edge_remap[unique_edges] = torch.arange(len(unique_edges),
                                                device=node_features.device)
        new_edge_idx = edge_remap[new_edge_idx]

        sub_he = torch.stack([new_node_idx, new_edge_idx])
        return sub_features, sub_he, K, len(unique_edges)

    def _get_subgraph_prediction(
        self,
        node_features: torch.Tensor,
        hyperedge_index: torch.Tensor,
        node_mask: torch.Tensor,
        num_edges: int,
        clinical_features: torch.Tensor = None,
    ) -> torch.Tensor:

        sub_feats, sub_he, sub_N, sub_E = self._extract_subgraph(
            node_features, hyperedge_index, node_mask, num_edges,
        )
        return self._get_prediction(
            sub_feats, sub_he, sub_N, sub_E, clinical_features,
        )

    def compute_est(
        self,
        node_features: torch.Tensor,
        hyperedge_index: torch.Tensor,
        num_nodes: int,
        num_edges: int,
        explanation_mask: torch.Tensor,
        full_prediction: torch.Tensor,
        clinical_features: torch.Tensor = None,
    ) -> Dict[str, float]:

        complement_mask = ~explanation_mask
        complement_indices = torch.where(complement_mask)[0]
        n_complement = len(complement_indices)

        if n_complement == 0:
            return {"est_score": 0.0, "est_pass": True}

        expl_pred = self._get_subgraph_prediction(
            node_features, hyperedge_index, explanation_mask,
            num_edges, clinical_features,
        )

        max_shift = 0.0

        for _ in range(self.est_samples):

            sample_size = torch.randint(1, max(2, n_complement), (1,)).item()
            perm = torch.randperm(n_complement)[:sample_size]
            sampled_complement = complement_indices[perm]

            supergraph_mask = explanation_mask.clone()
            supergraph_mask[sampled_complement] = True

            supergraph_pred = self._get_subgraph_prediction(
                node_features, hyperedge_index, supergraph_mask,
                num_edges, clinical_features,
            )

            shift = self._prediction_shift(expl_pred, supergraph_pred)
            max_shift = max(max_shift, shift)

        return {
            "est_score": max_shift,
            "est_pass": max_shift < self.threshold,
        }

    def compute_fid_minus(
        self,
        node_features: torch.Tensor,
        hyperedge_index: torch.Tensor,
        num_nodes: int,
        num_edges: int,
        explanation_mask: torch.Tensor,
        full_prediction: torch.Tensor,
        clinical_features: torch.Tensor = None,
    ) -> Dict[str, float]:

        expl_pred = self._get_subgraph_prediction(
            node_features, hyperedge_index, explanation_mask,
            num_edges, clinical_features,
        )

        shift = self._prediction_shift(full_prediction, expl_pred)

        return {
            "fid_minus_score": shift,
            "fid_minus_pass": shift < self.threshold,
        }

    def compute_rfid_minus(
        self,
        node_features: torch.Tensor,
        hyperedge_index: torch.Tensor,
        num_nodes: int,
        num_edges: int,
        explanation_mask: torch.Tensor,
        full_prediction: torch.Tensor,
        clinical_features: torch.Tensor = None,
    ) -> Dict[str, float]:

        complement_mask = ~explanation_mask

        shifts = []
        for _ in range(min(self.est_samples, 20)):

            complement_indices = torch.where(complement_mask)[0]
            survive = torch.rand(len(complement_indices),
                                 device=node_features.device) >= self.rfid_p

            perturbed_mask = explanation_mask.clone()
            perturbed_mask[complement_indices[survive]] = True

            pred = self._get_subgraph_prediction(
                node_features, hyperedge_index, perturbed_mask,
                num_edges, clinical_features,
            )
            shifts.append(self._prediction_shift(full_prediction, pred))

        mean_shift = np.mean(shifts) if shifts else 0.0

        return {
            "rfid_minus_score": mean_shift,
            "rfid_minus_pass": mean_shift < self.threshold,
        }

    def compute_sufficiency(
        self,
        node_features: torch.Tensor,
        hyperedge_index: torch.Tensor,
        num_nodes: int,
        num_edges: int,
        explanation_mask: torch.Tensor,
        full_prediction: torch.Tensor,
        other_node_features: torch.Tensor = None,
        clinical_features: torch.Tensor = None,
    ) -> Dict[str, float]:

        complement_mask = ~explanation_mask

        perturbed_features = node_features.clone()
        n_complement = complement_mask.sum().item()
        if n_complement > 0:
            noise = torch.randn_like(perturbed_features[complement_mask])

            feat_std = node_features.std()
            feat_mean = node_features.mean()
            noise = noise * feat_std + feat_mean
            perturbed_features[complement_mask] = noise

        pred = self._get_prediction(
            perturbed_features, hyperedge_index, num_nodes, num_edges,
            clinical_features,
        )

        shift = self._prediction_shift(full_prediction, pred)

        return {
            "sufficiency_score": shift,
            "sufficiency_pass": shift < self.threshold,
        }

    def audit_patient(
        self,
        data: Dict,
        explanation_mask: torch.Tensor = None,
        full_prediction: torch.Tensor = None,
        top_k_ratio: float = 0.2,
    ) -> Dict:

        self.model.eval()
        device = self.device

        node_features = data["node_features"].to(device)
        hyperedge_index = data["hyperedge_index"].to(device)
        num_nodes = data["num_nodes"]
        num_edges = data["num_hyperedges"]
        clinical = data["clinical_features"].to(device)

        if explanation_mask is None:
            extractor = ExplanationExtractor(top_k_ratio)
            expl_data = extractor.extract(
                self.model, node_features, hyperedge_index,
                num_nodes, num_edges,
                data.get("concepts", None),
                clinical,
            )
            explanation_mask = expl_data["explanation_mask"]
            full_prediction = expl_data["full_prediction"]
        elif full_prediction is None:
            full_prediction = self._get_prediction(
                node_features, hyperedge_index, num_nodes, num_edges, clinical,
            )

        explanation_mask = explanation_mask.to(device)
        full_prediction = full_prediction.to(device)

        est = self.compute_est(
            node_features, hyperedge_index, num_nodes, num_edges,
            explanation_mask, full_prediction, clinical,
        )
        fid = self.compute_fid_minus(
            node_features, hyperedge_index, num_nodes, num_edges,
            explanation_mask, full_prediction, clinical,
        )
        rfid = self.compute_rfid_minus(
            node_features, hyperedge_index, num_nodes, num_edges,
            explanation_mask, full_prediction, clinical,
        )
        suf = self.compute_sufficiency(
            node_features, hyperedge_index, num_nodes, num_edges,
            explanation_mask, full_prediction, clinical_features=clinical,
        )

        n_explanation = explanation_mask.sum().item()
        n_total = num_nodes

        return {
            "patient_id": data.get("patient_id", "unknown"),
            "num_nodes": n_total,
            "num_explanation_nodes": n_explanation,
            "explanation_ratio": n_explanation / max(n_total, 1),

            "est": est,
            "fid_minus": fid,
            "rfid_minus": rfid,
            "sufficiency": suf,

            "all_pass": all([
                est["est_pass"],
                fid["fid_minus_pass"],
                rfid["rfid_minus_pass"],
                suf["sufficiency_pass"],
            ]),
        }

def compute_rejection_ratios(
    audit_reports: List[Dict],
) -> Dict[str, float]:

    if not audit_reports:
        return {}

    n = len(audit_reports)

    est_fails = sum(1 for r in audit_reports if not r["est"]["est_pass"])
    fid_fails = sum(1 for r in audit_reports if not r["fid_minus"]["fid_minus_pass"])
    rfid_fails = sum(1 for r in audit_reports if not r["rfid_minus"]["rfid_minus_pass"])
    suf_fails = sum(1 for r in audit_reports if not r["sufficiency"]["sufficiency_pass"])
    all_fails = sum(1 for r in audit_reports if not r["all_pass"])

    return {
        "est_rejection": est_fails / n,
        "fid_minus_rejection": fid_fails / n,
        "rfid_minus_rejection": rfid_fails / n,
        "sufficiency_rejection": suf_fails / n,
        "overall_rejection": all_fails / n,
        "n_patients": n,

        "mean_est": np.mean([r["est"]["est_score"] for r in audit_reports]),
        "mean_fid_minus": np.mean([r["fid_minus"]["fid_minus_score"] for r in audit_reports]),
        "mean_rfid_minus": np.mean([r["rfid_minus"]["rfid_minus_score"] for r in audit_reports]),
        "mean_sufficiency": np.mean([r["sufficiency"]["sufficiency_score"] for r in audit_reports]),
    }
