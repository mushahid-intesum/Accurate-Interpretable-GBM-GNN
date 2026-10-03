import os
import sys
import json
import torch
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import PROCESSED_DIR, DEVICE
from data.dataset import GBMDataset

from explainability.random_explainer import RandomExplainer
from explainability.gradient_explainer import GradientExplainer
from explainability.ig_explainer import IntegratedGradientsExplainer
from explainability.attention_explainer import AttentionExplainer
from explainability.gnn_explainer import HypergraphGNNExplainer
from explainability.cbm_explainer import CBMExplainer

from explainability.faithfulness import UnifiedFaithfulnessAudit

def _load_model(checkpoint_path=None, device="cpu", config=None):
    from model.full_model import GBMModel

    if config is None:
        config = dict(
            patch_dim=1536, embed_dim=64, num_layers=3,
            clinical_dim=18, num_survival_bins=4,
            use_hecrl=True, use_fusion=True,
            residual_bypass=False, use_tree=False,
        )

    model = GBMModel(**config)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt)

    model = model.to(device)
    model.eval()
    return model

def build_explainers(model, device="cpu", top_k_ratio=0.2,
                     gnn_explainer_epochs=200, ig_steps=50):

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

    if processed_dir is None:
        processed_dir = str(PROCESSED_DIR)
    if output_dir is None:
        output_dir = str(Path(__file__).resolve().parent.parent / "checkpoints")

    print("=" * 75)
    print("  EXPLAINABILITY: Faithfulness Comparison")
    print("=" * 75)

    ds = GBMDataset(processed_dir, build_hypergraph=True)
    print(f"  Dataset: {len(ds.patient_ids)} patients")

    config = dict(
        patch_dim=1536, embed_dim=64, num_layers=3,
        clinical_dim=18, num_survival_bins=4,
        use_hecrl=True, use_fusion=True,
        residual_bypass=False, use_tree=False,
    )

    model_e6 = _load_model(checkpoint_e6, device, config)
    explainers = build_explainers(
        model_e6, device, top_k_ratio,
        gnn_explainer_epochs, ig_steps,
    )

    if checkpoint_e3 and os.path.exists(checkpoint_e3):
        config = dict(
            patch_dim=1536, embed_dim=64, num_layers=3,
            clinical_dim=18, num_survival_bins=4,
            use_hecrl=True, use_fusion=False,
            residual_bypass=False, use_tree=False,
        )

        model_e3 = _load_model(checkpoint_e3, device, config)
        explainers["CBM (E3)"] = CBMExplainer(model_e3, top_k_ratio, device)

        if "CBM" in explainers:
            explainers["CBM+EST (E6)"] = explainers.pop("CBM")

    audit = UnifiedFaithfulnessAudit(
        model_e6, explainers, device,
        est_samples=est_samples,
    )
    results = audit.audit_cohort(ds, n_patients=n_patients)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "faithfulness_comparison.json")

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

    from explainability.intervention import ConceptIntervenor

    if processed_dir is None:
        processed_dir = str(PROCESSED_DIR)
    if output_dir is None:
        output_dir = str(Path(__file__).resolve().parent.parent / "checkpoints")

    print("=" * 75)
    print("  EXPLAINABILITY: Concept Intervention")
    print("=" * 75)

    ds = GBMDataset(processed_dir, build_hypergraph=True)
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

    from explainability.analysis import ConceptClinicalAnalyzer

    if processed_dir is None:
        processed_dir = str(PROCESSED_DIR)
    if output_dir is None:
        output_dir = str(Path(__file__).resolve().parent.parent / "checkpoints")

    print("=" * 75)
    print("  EXPLAINABILITY: Clinical Concept Analysis")
    print("=" * 75)

    ds = GBMDataset(processed_dir, build_hypergraph=True)
    model = _load_model(checkpoint_e6, device)
    analyzer = ConceptClinicalAnalyzer(model, device)
    results = analyzer.analyze_cohort(ds, n_patients=n_patients)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "concept_analysis.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")
    return results

if __name__ == "__main__":
    EXPERIMENT_FAITHFULNESS = "faithfulness"
    EXPERIMENT_INTERVENTION = "intervention"
    EXPERIMENT_ANALYSIS = "analysis"
    EXPERIMENT_ALL = "all"

    EXPERIMENT = EXPERIMENT_ALL

    PROCESSED_DIR_ = None
    CHECKPOINT_E6 = '/mnt/Stuff/arche/arche-brain-tumor-gnn/checkpoints/E6_fold4_best.pt'
    CHECKPOINT_E3 = '/mnt/Stuff/arche/arche-brain-tumor-gnn/checkpoints/E3_fold4_best.pt'
    N_PATIENTS = 200
    DEVICE = "cuda"
    TOP_K = 0.2
    GNN_EPOCHS = 200
    IG_STEPS = 50
    EST_SAMPLES = 50

    if EXPERIMENT in (EXPERIMENT_INTERVENTION, EXPERIMENT_ALL):
        run_intervention(
            processed_dir=PROCESSED_DIR_,
            checkpoint_e6=CHECKPOINT_E6,
            n_patients=N_PATIENTS,
            device=DEVICE,
        )

    if EXPERIMENT in (EXPERIMENT_ANALYSIS, EXPERIMENT_ALL):
        run_analysis(
            processed_dir=PROCESSED_DIR,
            checkpoint_e6=CHECKPOINT_E6,
            n_patients=N_PATIENTS,
            device=DEVICE,
        )
