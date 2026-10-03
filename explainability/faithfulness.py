import numpy as np
import torch
from typing import Dict, List, Optional
from collections import defaultdict

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from plan3a.eval.faithfulness import FaithfulnessAuditor
from plan3a.explainability.base import BaseExplainer

class UnifiedFaithfulnessAudit:

    def __init__(
        self,
        model,
        explainers: Dict[str, BaseExplainer],
        device: str = "cpu",
        est_samples: int = 50,
        prediction_threshold: float = 0.1,
    ):

        self.model = model
        self.explainers = explainers
        self.device = device
        self.auditor = FaithfulnessAuditor(
            model,
            device=device,
            est_samples=est_samples,
            prediction_threshold=prediction_threshold,
        )

    def audit_patient_all_methods(
        self, patient_data: Dict,
    ) -> Dict[str, Dict]:

        results = {}

        for name, explainer in self.explainers.items():

            explanation = explainer.explain(patient_data)

            audit = self.auditor.audit_patient(
                patient_data,
                explanation_mask=explanation["explanation_mask"],
                full_prediction=explanation["full_prediction"],
            )

            audit["method"] = name
            audit["metadata"] = explanation.get("metadata", {})
            results[name] = audit

        return results

    def audit_cohort(
        self,
        dataset,
        n_patients: int = 25,
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
            print(f"Auditing {len(valid_indices)} patients with "
                  f"{len(self.explainers)} methods...")

        per_patient = []
        method_scores = defaultdict(lambda: defaultdict(list))

        for idx_num, idx in enumerate(valid_indices):
            patient_data = dataset[idx]
            pid = patient_data.get("patient_id", f"patient_{idx}")

            if verbose:
                print(f"  [{idx_num+1}/{len(valid_indices)}] {pid} "
                      f"({patient_data['num_nodes']} nodes)", end="")

            patient_results = self.audit_patient_all_methods(patient_data)

            if verbose:

                parts = []
                for name, audit in patient_results.items():
                    est = audit["est"]["est_score"]
                    parts.append(f"{name}:{est:.3f}")
                print(f"  EST=[{', '.join(parts)}]")

            per_patient.append({
                "patient_id": pid,
                "results": patient_results,
            })

            for name, audit in patient_results.items():
                method_scores[name]["est"].append(
                    audit["est"]["est_score"])
                method_scores[name]["est_pass"].append(
                    audit["est"]["est_pass"])
                method_scores[name]["fid_minus"].append(
                    audit["fid_minus"]["fid_minus_score"])
                method_scores[name]["fid_minus_pass"].append(
                    audit["fid_minus"]["fid_minus_pass"])
                method_scores[name]["rfid_minus"].append(
                    audit["rfid_minus"]["rfid_minus_score"])
                method_scores[name]["rfid_minus_pass"].append(
                    audit["rfid_minus"]["rfid_minus_pass"])
                method_scores[name]["sufficiency"].append(
                    audit["sufficiency"]["sufficiency_score"])
                method_scores[name]["sufficiency_pass"].append(
                    audit["sufficiency"]["sufficiency_pass"])

        summary = {}
        rejection_ratios = {}

        for name in self.explainers:
            scores = method_scores[name]
            n = len(scores["est"])

            summary[name] = {
                "mean_est": float(np.mean(scores["est"])),
                "std_est": float(np.std(scores["est"])),
                "mean_fid_minus": float(np.mean(scores["fid_minus"])),
                "std_fid_minus": float(np.std(scores["fid_minus"])),
                "mean_rfid_minus": float(np.mean(scores["rfid_minus"])),
                "std_rfid_minus": float(np.std(scores["rfid_minus"])),
                "mean_sufficiency": float(np.mean(scores["sufficiency"])),
                "std_sufficiency": float(np.std(scores["sufficiency"])),
            }

            rejection_ratios[name] = {
                "est": 1.0 - np.mean(scores["est_pass"]),
                "fid_minus": 1.0 - np.mean(scores["fid_minus_pass"]),
                "rfid_minus": 1.0 - np.mean(scores["rfid_minus_pass"]),
                "sufficiency": 1.0 - np.mean(scores["sufficiency_pass"]),
                "overall": 1.0 - np.mean([
                    all(t) for t in zip(
                        scores["est_pass"],
                        scores["fid_minus_pass"],
                        scores["rfid_minus_pass"],
                        scores["sufficiency_pass"],
                    )
                ]),
                "n_patients": n,
            }

        if verbose:
            self._print_table(summary, rejection_ratios)

        return {
            "per_patient": per_patient,
            "summary": summary,
            "rejection_ratios": rejection_ratios,
            "n_patients": len(valid_indices),
        }

    @staticmethod
    def _print_table(summary: Dict, rejection_ratios: Dict):

        print("\n" + "=" * 75)
        print("  FAITHFULNESS COMPARISON TABLE")
        print("=" * 75)
        print(f"  {'Method':<20} {'EST↓':>8} {'Fid-↓':>8} "
              f"{'RFid-↓':>8} {'Suf↓':>8} {'Reject%':>8}")
        print("-" * 75)

        for name in summary:
            s = summary[name]
            r = rejection_ratios[name]
            print(
                f"  {name:<20} "
                f"{s['mean_est']:>7.4f}  "
                f"{s['mean_fid_minus']:>7.4f}  "
                f"{s['mean_rfid_minus']:>7.4f}  "
                f"{s['mean_sufficiency']:>7.4f}  "
                f"{r['overall']:>6.1%}"
            )

        print("=" * 75)
        print("  Lower scores = more faithful explanations")
        print("  Reject% = fraction of explanations failing any metric")
        print("=" * 75)
