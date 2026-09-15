"""
Explanation Analysis — alignment comparison + clinical concept analysis.

Two analysis modules:

1. ExplanationAligner: Compare ante-hoc (CBM) vs post-hoc (GNNExplainer)
   explanations to measure inter-method agreement.

2. ConceptClinicalAnalyzer: Correlate learned concepts with actual
   survival outcomes and clinical covariates to verify the model
   has learned clinically meaningful representations.
"""
import torch
import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict
from scipy import stats

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from plan3a.config import NUM_CONCEPTS
from plan3a.explainability.intervention import CONCEPT_NAMES


class ExplanationAligner:
    """
    Compare explanations from two methods (typically CBM vs GNNExplainer).

    Computes:
    1. Spearman rank correlation of node importance scores
    2. Top-k Jaccard overlap of explanation masks
    3. Per-concept node-importance correlation

    High alignment = both methods agree on which nodes matter.
    Disagreement = potential faithfulness issues in one or both methods.
    """

    def align_patient(
        self,
        importance_a: torch.Tensor,
        importance_b: torch.Tensor,
        mask_a: torch.Tensor,
        mask_b: torch.Tensor,
        concepts: Optional[torch.Tensor] = None,
    ) -> Dict:
        """
        Compare two explanations for one patient.

        Args:
            importance_a: (N,) node importance from method A
            importance_b: (N,) node importance from method B
            mask_a: (N,) boolean explanation mask from A
            mask_b: (N,) boolean explanation mask from B
            concepts: (N, 8) concept activations (optional)

        Returns:
            dict with alignment metrics
        """
        a_np = importance_a.cpu().numpy()
        b_np = importance_b.cpu().numpy()

        # Spearman rank correlation
        rho, pval = stats.spearmanr(a_np, b_np)

        # Jaccard overlap of top-k masks
        intersection = (mask_a & mask_b).sum().item()
        union = (mask_a | mask_b).sum().item()
        jaccard = intersection / max(union, 1)

        # Overlap metrics
        a_size = mask_a.sum().item()
        b_size = mask_b.sum().item()
        precision = intersection / max(a_size, 1)
        recall = intersection / max(b_size, 1)

        result = {
            "spearman_rho": float(rho) if not np.isnan(rho) else 0.0,
            "spearman_pval": float(pval) if not np.isnan(pval) else 1.0,
            "jaccard": jaccard,
            "precision": precision,
            "recall": recall,
            "mask_a_size": a_size,
            "mask_b_size": b_size,
            "intersection": intersection,
        }

        # Per-concept correlation (if concepts available)
        if concepts is not None:
            concept_cors = []
            for cidx in range(min(concepts.shape[1], NUM_CONCEPTS)):
                c_np = concepts[:, cidx].cpu().numpy()
                r_a, _ = stats.spearmanr(c_np, a_np)
                r_b, _ = stats.spearmanr(c_np, b_np)
                concept_cors.append({
                    "concept": CONCEPT_NAMES[cidx] if cidx < len(CONCEPT_NAMES) else f"c{cidx+1}",
                    "corr_with_a": float(r_a) if not np.isnan(r_a) else 0.0,
                    "corr_with_b": float(r_b) if not np.isnan(r_b) else 0.0,
                })
            result["concept_correlations"] = concept_cors

        return result

    def align_cohort(
        self,
        patient_results: List[Dict],
    ) -> Dict:
        """
        Aggregate alignment metrics across patients.
        """
        if not patient_results:
            return {}

        rhos = [r["spearman_rho"] for r in patient_results]
        jaccards = [r["jaccard"] for r in patient_results]

        return {
            "mean_spearman": float(np.mean(rhos)),
            "std_spearman": float(np.std(rhos)),
            "mean_jaccard": float(np.mean(jaccards)),
            "std_jaccard": float(np.std(jaccards)),
            "n_patients": len(patient_results),
        }


class ConceptClinicalAnalyzer:
    """
    Analyze clinical meaning and survival relevance of learned concepts.

    For each of the 8 concepts, computes:
    1. Distribution statistics (mean, std across cohort)
    2. Survival correlation (Spearman r with actual survival time)
    3. Clinical covariate correlation (with age, IDH1, MGMT, etc.)
    4. Inter-concept correlation matrix
    """

    CLINICAL_NAMES = [
        "Age", "KPS", "Gender", "Race",
        "IDH1", "MGMT", "1p19q", "ATRX",
        "EGFR", "TP53", "PTEN", "PIK3CA",
        "NF1", "RB1", "CDKN2A", "CDK4",
        "MDM2", "PDGFRA",
    ]

    def __init__(self, model, device: str = "cpu"):
        self.model = model
        self.device = device

    def analyze_cohort(
        self,
        dataset,
        n_patients: int = 100,
        verbose: bool = True,
    ) -> Dict:
        """
        Run clinical concept analysis across the cohort.

        Collects per-patient graph-level concept activations,
        survival times, events, and clinical features.
        """
        self.model.eval()

        # Collect data
        concept_matrix = []    # (P, 8) graph-level concept means
        survival_times = []
        events = []
        clinical_matrix = []

        valid = 0
        for i in range(len(dataset)):
            if valid >= n_patients:
                break

            patient = dataset[i]
            if not patient.get("has_survival", False):
                continue

            node_feats = patient["node_features"].to(self.device)
            hg = patient["hyperedge_index"].to(self.device)
            num_nodes = patient["num_nodes"]
            num_edges = patient["num_hyperedges"]
            concepts_t = patient.get("concepts")
            if concepts_t is not None:
                concepts_t = concepts_t.to(self.device)
            clinical = patient.get("clinical_features")
            if clinical is not None:
                clinical = clinical.to(self.device)

            with torch.no_grad():
                outputs = self.model(
                    node_features=node_feats,
                    hyperedge_index=hg,
                    num_nodes=num_nodes,
                    num_edges=num_edges,
                    concept_targets=concepts_t,
                    clinical_features=clinical,
                )

            # Graph-level concept = mean across nodes
            concepts = outputs["concepts"].mean(dim=0).cpu().numpy()  # (8,)
            concept_matrix.append(concepts)
            survival_times.append(patient["survival_time"].item())
            events.append(patient["event"].item())
            if clinical is not None:
                clinical_matrix.append(clinical.cpu().numpy())

            valid += 1

        if verbose:
            print(f"Analyzed {valid} patients")

        concept_matrix = np.array(concept_matrix)  # (P, 8)
        survival_times = np.array(survival_times)
        events = np.array(events)
        clinical_matrix = np.array(clinical_matrix) if clinical_matrix else None

        # ── 1. Distribution statistics ───────────────────────────
        distributions = []
        for cidx in range(NUM_CONCEPTS):
            vals = concept_matrix[:, cidx]
            distributions.append({
                "concept": CONCEPT_NAMES[cidx],
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "median": float(np.median(vals)),
            })

        # ── 2. Survival correlation ──────────────────────────────
        survival_corrs = []
        for cidx in range(NUM_CONCEPTS):
            rho, pval = stats.spearmanr(concept_matrix[:, cidx], survival_times)
            survival_corrs.append({
                "concept": CONCEPT_NAMES[cidx],
                "spearman_rho": float(rho) if not np.isnan(rho) else 0.0,
                "p_value": float(pval) if not np.isnan(pval) else 1.0,
                "significant": bool(pval < 0.05) if not np.isnan(pval) else False,
            })

        # ── 3. Clinical covariate correlation ────────────────────
        clinical_corrs = None
        if clinical_matrix is not None and clinical_matrix.shape[1] > 0:
            clinical_corrs = []
            n_clin = min(clinical_matrix.shape[1], len(self.CLINICAL_NAMES))
            for cidx in range(NUM_CONCEPTS):
                row = {"concept": CONCEPT_NAMES[cidx]}
                for vidx in range(n_clin):
                    rho, _ = stats.spearmanr(
                        concept_matrix[:, cidx], clinical_matrix[:, vidx],
                    )
                    row[self.CLINICAL_NAMES[vidx]] = (
                        float(rho) if not np.isnan(rho) else 0.0
                    )
                clinical_corrs.append(row)

        # ── 4. Inter-concept correlation matrix ──────────────────
        inter_concept = np.corrcoef(concept_matrix.T)  # (8, 8)

        result = {
            "n_patients": valid,
            "distributions": distributions,
            "survival_correlations": survival_corrs,
            "clinical_correlations": clinical_corrs,
            "inter_concept_matrix": inter_concept.tolist(),
        }

        if verbose:
            self._print_report(result)

        return result

    @staticmethod
    def _print_report(result: Dict):
        """Print clinical analysis report."""
        print(f"\n{'=' * 70}")
        print("  CONCEPT CLINICAL ANALYSIS")
        print(f"{'=' * 70}")

        # Distributions
        print(f"\n  {'Concept':<22} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
        print("-" * 60)
        for d in result["distributions"]:
            print(f"  {d['concept']:<22} {d['mean']:>8.3f} {d['std']:>8.3f} "
                  f"{d['min']:>8.3f} {d['max']:>8.3f}")

        # Survival correlations
        print(f"\n  {'Concept':<22} {'ρ(survival)':>12} {'p-value':>10} {'Sig?':>6}")
        print("-" * 55)
        for s in result["survival_correlations"]:
            sig = "***" if s["p_value"] < 0.001 else (
                "**" if s["p_value"] < 0.01 else (
                    "*" if s["p_value"] < 0.05 else ""))
            print(f"  {s['concept']:<22} {s['spearman_rho']:>+11.4f} "
                  f"{s['p_value']:>10.4f} {sig:>6}")

        # Inter-concept
        print(f"\n  Inter-concept correlation matrix (Pearson r):")
        names_short = [f"c{i+1}" for i in range(NUM_CONCEPTS)]
        print(f"  {'':>6}", end="")
        for n in names_short:
            print(f" {n:>6}", end="")
        print()
        mat = result["inter_concept_matrix"]
        for i, name in enumerate(names_short):
            print(f"  {name:>6}", end="")
            for j in range(len(names_short)):
                print(f" {mat[i][j]:>+5.2f}", end="")
            print()

        print(f"\n{'=' * 70}")
