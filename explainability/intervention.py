import torch
import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import NUM_CONCEPTS

CONCEPT_NAMES = [
    "c1: Enhancement",
    "c2: FLAIR/Edema",
    "c3: T2 Abnormality",
    "c4: DTI MD",
    "c5: DTI FA",
    "c6: Heterogeneity",
    "c7: Boundary",
    "c8: Spatial Loc",
]

def _compute_risk(hazard_logits: torch.Tensor) -> float:

    hazard = torch.sigmoid(hazard_logits)
    survival = torch.prod(1.0 - hazard, dim=-1)
    return float(1.0 - survival)

class ConceptIntervenor:

    def __init__(self, model, device: str = "cpu"):
        self.model = model
        self.device = device

    def _run_stages(
        self,
        patient_data: Dict,
        concept_override: Optional[Dict] = None,
        intervention_point: str = "pre_hecrl",
    ) -> Dict[str, torch.Tensor]:

        self.model.eval()

        node_feats = patient_data["node_features"].to(self.device)
        hg = patient_data["hyperedge_index"].to(self.device)
        num_nodes = patient_data["num_nodes"]
        num_edges = patient_data["num_hyperedges"]
        concepts_target = patient_data.get("concepts")
        if concepts_target is not None:
            concepts_target = concepts_target.to(self.device)
        clinical = patient_data.get("clinical_features")
        if clinical is not None:
            clinical = clinical.to(self.device)

        with torch.no_grad():

            node_embeds = self.model.shgnn(
                node_feats, hg, num_nodes, num_edges,
            )

            concepts_raw = self.model.concept_bottleneck.predictor(node_embeds)

            boundary = self.model.concept_bottleneck.boundary_head(node_embeds)
            concepts_raw = concepts_raw.clone()
            concepts_raw[:, 6] = boundary.squeeze(-1)

            if concept_override and intervention_point == "pre_hecrl":
                concepts_raw = concepts_raw.clone()
                for cidx, new_val in concept_override.items():
                    concepts_raw[:, cidx] = new_val

            if self.model.concept_bottleneck.use_hecrl:
                concepts_refined = self.model.concept_bottleneck.hecrl(concepts_raw)
            else:
                concepts_refined = concepts_raw

            if concept_override and intervention_point == "post_hecrl":
                concepts_refined = concepts_refined.clone()
                for cidx, new_val in concept_override.items():
                    concepts_refined[:, cidx] = new_val

            bottleneck_output = concepts_refined

            concept_graph = self.model.concept_pool(bottleneck_output)
            graph_embed = self.model.pooler(concept_graph)

            if self.model.use_fusion and clinical is not None:
                if clinical.dim() == 1:
                    clinical = clinical.unsqueeze(0)
                fusion_out = self.model.fusion(graph_embed, clinical)
                fused = fusion_out["fused"]
            else:
                fused = graph_embed

            hazard_logits = self.model.survival_head(fused)

        return {
            "hazard_logits": hazard_logits,
            "concepts_raw": concepts_raw,
            "concepts_refined": concepts_refined,
            "risk": _compute_risk(hazard_logits),
        }

    def intervene_single(
        self,
        patient_data: Dict,
        concept_idx: int,
        new_value: float,
        intervention_point: str = "pre_hecrl",
    ) -> Dict:

        orig = self._run_stages(patient_data)

        interv = self._run_stages(
            patient_data,
            concept_override={concept_idx: new_value},
            intervention_point=intervention_point,
        )

        return {
            "concept_idx": concept_idx,
            "concept_name": CONCEPT_NAMES[concept_idx],
            "new_value": new_value,
            "intervention_point": intervention_point,
            "original_risk": orig["risk"],
            "intervened_risk": interv["risk"],
            "delta_risk": interv["risk"] - orig["risk"],
        }

    def sweep_all_concepts(
        self,
        patient_data: Dict,
        intervention_point: str = "pre_hecrl",
        population_stats: Optional[Dict] = None,
    ) -> Dict:

        concepts_target = patient_data.get("concepts")
        if concepts_target is not None:

            gt_values = concepts_target.mean(dim=0).tolist()
        else:
            gt_values = [0.0] * NUM_CONCEPTS

        max_values = [3.0] * NUM_CONCEPTS
        if population_stats and "max" in population_stats:
            max_values = population_stats["max"]

        results = []
        for cidx in range(NUM_CONCEPTS):
            row = {"concept_idx": cidx, "concept_name": CONCEPT_NAMES[cidx]}

            r = self.intervene_single(
                patient_data, cidx, 0.0, intervention_point,
            )
            row["zero_out"] = r["delta_risk"]

            r = self.intervene_single(
                patient_data, cidx, max_values[cidx], intervention_point,
            )
            row["maximize"] = r["delta_risk"]

            r = self.intervene_single(
                patient_data, cidx, gt_values[cidx], intervention_point,
            )
            row["gt_correct"] = r["delta_risk"]

            results.append(row)

        return {
            "patient_id": patient_data.get("patient_id", "unknown"),
            "intervention_point": intervention_point,
            "concept_results": results,
        }

    def cohort_analysis(
        self,
        dataset,
        n_patients: int = 50,
        verbose: bool = True,
    ) -> Dict:

        valid_indices = []
        for i in range(len(dataset)):
            d = dataset[i]
            if d.get("has_survival", False):
                valid_indices.append(i)
            if len(valid_indices) >= n_patients:
                break

        if verbose:
            print(f"Running concept intervention on {len(valid_indices)} patients...")

        all_results = {"pre_hecrl": [], "post_hecrl": []}

        for idx_num, idx in enumerate(valid_indices):
            patient_data = dataset[idx]
            pid = patient_data.get("patient_id", f"patient_{idx}")

            if verbose and (idx_num + 1) % 10 == 0:
                print(f"  [{idx_num+1}/{len(valid_indices)}] {pid}")

            for point in ["pre_hecrl", "post_hecrl"]:
                result = self.sweep_all_concepts(patient_data, point)
                all_results[point].append(result)

        aggregated = {}
        for point in ["pre_hecrl", "post_hecrl"]:
            concept_deltas = defaultdict(lambda: {"zero": [], "max": [], "gt": []})

            for patient_result in all_results[point]:
                for cr in patient_result["concept_results"]:
                    cidx = cr["concept_idx"]
                    concept_deltas[cidx]["zero"].append(cr["zero_out"])
                    concept_deltas[cidx]["max"].append(cr["maximize"])
                    concept_deltas[cidx]["gt"].append(cr["gt_correct"])

            point_summary = []
            for cidx in range(NUM_CONCEPTS):
                d = concept_deltas[cidx]
                point_summary.append({
                    "concept_idx": cidx,
                    "concept_name": CONCEPT_NAMES[cidx],
                    "mean_zero_delta": float(np.mean(d["zero"])),
                    "mean_max_delta": float(np.mean(d["max"])),
                    "mean_gt_delta": float(np.mean(d["gt"])),
                    "std_zero_delta": float(np.std(d["zero"])),
                    "std_max_delta": float(np.std(d["max"])),
                    "std_gt_delta": float(np.std(d["gt"])),
                })

            aggregated[point] = point_summary

        if verbose:
            self._print_intervention_table(aggregated)

        return {
            "n_patients": len(valid_indices),
            "aggregated": aggregated,
            "per_patient": all_results,
        }

    @staticmethod
    def _print_intervention_table(aggregated: Dict):

        for point in ["pre_hecrl", "post_hecrl"]:
            print(f"\n{'=' * 70}")
            print(f"  CONCEPT INTERVENTION ({point.upper()})")
            print(f"{'=' * 70}")
            print(f"  {'Concept':<22} {'Zero Δrisk':>12} {'Max Δrisk':>12} {'GT Δrisk':>12}")
            print("-" * 70)

            for row in aggregated[point]:
                print(
                    f"  {row['concept_name']:<22} "
                    f"{row['mean_zero_delta']:>+11.4f}  "
                    f"{row['mean_max_delta']:>+11.4f}  "
                    f"{row['mean_gt_delta']:>+11.4f}"
                )

            print("=" * 70)
