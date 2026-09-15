"""
Explainability Runner — CLI for all explainability experiments.

Usage:
    python -m plan3a.explainability.runner --experiment faithfulness
    python -m plan3a.explainability.runner --experiment faithfulness --n-patients 5
    python -m plan3a.explainability.runner --experiment intervention
    python -m plan3a.explainability.runner --experiment all

Requires a trained Plan3aModel. For smoke testing, uses an untrained
model (results will be meaningless but the pipeline is validated).

Configuration is controlled via plan3a.config constants.
"""
import os
import sys
import json
import torch
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from plan3a.config import PROCESSED_DIR, DEVICE
from plan3a.data.dataset import Plan3aDataset

# Explainers
from plan3a.explainability.random_explainer import RandomExplainer
from plan3a.explainability.gradient_explainer import GradientExplainer
from plan3a.explainability.ig_explainer import IntegratedGradientsExplainer
from plan3a.explainability.attention_explainer import AttentionExplainer
from plan3a.explainability.gnn_explainer import HypergraphGNNExplainer
from plan3a.explainability.cbm_explainer import CBMExplainer

# Audit
from plan3a.explainability.faithfulness import UnifiedFaithfulnessAudit


def _load_model(checkpoint_path=None, device="cpu"):
    """
    Load a trained Plan3aModel.

    If no checkpoint is provided, creates an untrained model
    (for smoke testing the pipeline).
    """
    from plan3a.model.full_model import Plan3aModel

    model = Plan3aModel(
        patch_dim=1536, embed_dim=64, num_layers=3,
        clinical_dim=18, num_survival_bins=4,
        use_hecrl=True, use_fusion=True,
    )

    if checkpoint_path and os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=device)
        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        else:
            model.load_state_dict(state)
        print(f"  Loaded checkpoint: {checkpoint_path}")
    else:
        print(f"  WARNING: No checkpoint loaded, using untrained model")

    model = model.to(device)
    model.eval()
    return model


def build_explainers(model, device="cpu", top_k_ratio=0.2,
                     gnn_explainer_epochs=200, ig_steps=50):
    """
    Create all 7 explainer instances (or subset for two-model comparison).

    For single-model testing, CBM is included once.
    For E3 vs E6 comparison, call this separately for each model.
    """
    return {
        "Random": RandomExplainer(model, top_k_ratio, device),
        "Gradient": GradientExplainer(model, top_k_ratio, device),
        "IntGrad": IntegratedGradientsExplainer(
            model, top_k_ratio, device, n_steps=ig_steps,
        ),
        "Attention": AttentionExplainer(model, top_k_ratio, device),
        "GNNExplainer": HypergraphGNNExplainer(
            model, top_k_ratio, device,
            optim_epochs=gnn_explainer_epochs,
        ),
        "CBM": CBMExplainer(model, top_k_ratio, device),
    }


def run_faithfulness_comparison(
    processed_dir=None,
    checkpoint_e6=None,
    checkpoint_e3=None,
    n_patients=25,
    device="cpu",
    top_k_ratio=0.2,
    gnn_explainer_epochs=200,
    ig_steps=50,
    est_samples=50,
    output_dir=None,
):
    """
    Run faithfulness comparison across all explanation methods.

    If both E3 and E6 checkpoints are provided, produces the full
    7-row table. Otherwise, produces a 6-row table with one model.

    Returns:
        dict with comparison results
    """
    if processed_dir is None:
        processed_dir = str(PROCESSED_DIR)
    if output_dir is None:
        output_dir = str(Path(__file__).resolve().parent.parent / "checkpoints")

    print("=" * 75)
    print("  EXPLAINABILITY: Faithfulness Comparison")
    print("=" * 75)

    # Load dataset
    ds = Plan3aDataset(processed_dir, build_hypergraph=True)
    print(f"  Dataset: {len(ds.patient_ids)} patients")

    # Load model(s)
    model_e6 = _load_model(checkpoint_e6, device)
    explainers = build_explainers(
        model_e6, device, top_k_ratio,
        gnn_explainer_epochs, ig_steps,
    )

    # If E3 checkpoint available, add CBM(E3) as separate entry
    if checkpoint_e3 and os.path.exists(checkpoint_e3):
        model_e3 = _load_model(checkpoint_e3, device)
        explainers["CBM (E3)"] = CBMExplainer(model_e3, top_k_ratio, device)
        # Rename existing CBM to CBM+EST
        if "CBM" in explainers:
            explainers["CBM+EST (E6)"] = explainers.pop("CBM")

    # Run audit
    audit = UnifiedFaithfulnessAudit(
        model_e6, explainers, device,
        est_samples=est_samples,
    )
    results = audit.audit_cohort(ds, n_patients=n_patients)

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "faithfulness_comparison.json")

    # Make results JSON-serializable
    save_data = {
        "experiment": "faithfulness_comparison",
        "timestamp": datetime.now().isoformat(),
        "n_patients": results["n_patients"],
        "top_k_ratio": top_k_ratio,
        "est_samples": est_samples,
        "summary": results["summary"],
        "rejection_ratios": {
            k: {kk: float(vv) for kk, vv in v.items()}
            for k, v in results["rejection_ratios"].items()
        },
    }

    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")

    return results


def run_intervention(
    processed_dir=None,
    checkpoint_e6=None,
    n_patients=50,
    device="cpu",
    output_dir=None,
):
    """Run concept intervention experiment."""
    from plan3a.explainability.intervention import ConceptIntervenor

    if processed_dir is None:
        processed_dir = str(PROCESSED_DIR)
    if output_dir is None:
        output_dir = str(Path(__file__).resolve().parent.parent / "checkpoints")

    print("=" * 75)
    print("  EXPLAINABILITY: Concept Intervention")
    print("=" * 75)

    ds = Plan3aDataset(processed_dir, build_hypergraph=True)
    model = _load_model(checkpoint_e6, device)
    intervenor = ConceptIntervenor(model, device)
    results = intervenor.cohort_analysis(ds, n_patients=n_patients)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "intervention_results.json")
    save_data = {
        "experiment": "concept_intervention",
        "timestamp": datetime.now().isoformat(),
        "n_patients": results["n_patients"],
        "aggregated": results["aggregated"],
    }
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")
    return results


def run_analysis(
    processed_dir=None,
    checkpoint_e6=None,
    n_patients=100,
    device="cpu",
    output_dir=None,
):
    """Run clinical concept analysis."""
    from plan3a.explainability.analysis import ConceptClinicalAnalyzer

    if processed_dir is None:
        processed_dir = str(PROCESSED_DIR)
    if output_dir is None:
        output_dir = str(Path(__file__).resolve().parent.parent / "checkpoints")

    print("=" * 75)
    print("  EXPLAINABILITY: Clinical Concept Analysis")
    print("=" * 75)

    ds = Plan3aDataset(processed_dir, build_hypergraph=True)
    model = _load_model(checkpoint_e6, device)
    analyzer = ConceptClinicalAnalyzer(model, device)
    results = analyzer.analyze_cohort(ds, n_patients=n_patients)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "concept_analysis.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")
    return results


# ── CLI ──────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run explainability experiments"
    )
    parser.add_argument(
        "--experiment", type=str, required=True,
        choices=["faithfulness", "intervention", "analysis", "all"],
        help="Which experiment to run",
    )
    parser.add_argument("--processed-dir", type=str, default=None)
    parser.add_argument("--checkpoint-e6", type=str, default=None,
                        help="Path to E6 (CBM+EST) checkpoint")
    parser.add_argument("--checkpoint-e3", type=str, default=None,
                        help="Path to E3 (CBM, no EST) checkpoint")
    parser.add_argument("--n-patients", type=int, default=25)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--top-k", type=float, default=0.2)
    parser.add_argument("--gnn-epochs", type=int, default=200)
    parser.add_argument("--ig-steps", type=int, default=50)
    parser.add_argument("--est-samples", type=int, default=50)
    args = parser.parse_args()

    if args.experiment in ("faithfulness", "all"):
        run_faithfulness_comparison(
            processed_dir=args.processed_dir,
            checkpoint_e6=args.checkpoint_e6,
            checkpoint_e3=args.checkpoint_e3,
            n_patients=args.n_patients,
            device=args.device,
            top_k_ratio=args.top_k,
            gnn_explainer_epochs=args.gnn_epochs,
            ig_steps=args.ig_steps,
            est_samples=args.est_samples,
        )

    if args.experiment in ("intervention", "all"):
        run_intervention(
            processed_dir=args.processed_dir,
            checkpoint_e6=args.checkpoint_e6,
            n_patients=args.n_patients,
            device=args.device,
        )

    if args.experiment in ("analysis", "all"):
        run_analysis(
            processed_dir=args.processed_dir,
            checkpoint_e6=args.checkpoint_e6,
            n_patients=args.n_patients,
            device=args.device,
        )
