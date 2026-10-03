import os, sys, json, argparse
import numpy as np
import torch
from pathlib import Path
from scipy.stats import wilcoxon, ttest_rel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sksurv.metrics import (
    cumulative_dynamic_auc,
    integrated_brier_score,
)
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

from plan3a.config import PROCESSED_DIR, DEVICE, NUM_FOLDS, SHEAF_HGNN_DIM, SHEAF_HGNN_LAYERS, NUM_CONCEPTS
from plan3a.data.dataset import Plan3aDataset, get_kfold_splits
from plan3a.model.full_model import Plan3aModel
from plan3a.eval.task_metrics import concordance_index, compute_time_bins, hazard_to_risk
from plan3a.runner import EXPERIMENTS, HypergraphDatasetWrapper, build_knn_graph

def _load_model(exp_id, fold_idx, checkpoints_dir, device):

    exp_config = EXPERIMENTS[exp_id]
    model = Plan3aModel(
        patch_dim=1536,
        embed_dim=SHEAF_HGNN_DIM,
        num_layers=SHEAF_HGNN_LAYERS,
        num_concepts=NUM_CONCEPTS,
        clinical_dim=18,
        num_survival_bins=4,
        **exp_config["model_kwargs"],
    ).to(device)

    ckpt_path = os.path.join(checkpoints_dir, f"{exp_id}_fold{fold_idx}_best.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()
    return model

def _load_baseline_model(model_name, fold_idx, checkpoints_dir, device):

    ckpt_path = os.path.join(checkpoints_dir, f"{model_name}_fold{fold_idx}_best.pt")
    if not os.path.exists(ckpt_path):
        return None

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = state.get("model_state_dict", state)

    if model_name == "deepsurv":
        from plan3a.baselines.deepsurv import DeepSurvWithNLL
        model = DeepSurvWithNLL(in_dim=18, num_bins=4).to(device)
    elif model_name == "hypercbm":
        from plan3a.baselines.hypercbm import StandaloneHyperCBM
        model = StandaloneHyperCBM().to(device)
    elif model_name == "mrepath":
        from plan3a.baselines.mrepath import StandaloneMRePath
        model = StandaloneMRePath().to(device)
    else:
        raise ValueError(f"Unknown baseline: {model_name}")

    model.load_state_dict(sd)
    model.eval()
    return model

@torch.no_grad()
def _get_patient_risks_main(model, dataset, device):

    risks, times, events, pids = [], [], [], []
    for i in range(len(dataset)):
        data = dataset[i]
        if not data.get("has_survival", False):
            continue
        nf = data["node_features"].to(device)
        he = data["hyperedge_index"].to(device)
        nn_ = data["num_nodes"]
        ne = data["num_hyperedges"]
        clin = data["clinical_features"].to(device)
        ct = data["concepts"].to(device) if "concepts" in data else None

        out = model(nf, he, nn_, ne, concept_targets=ct, clinical_features=clin)
        hazard = torch.sigmoid(out["hazard_logits"]).cpu().numpy()
        risk = float(hazard_to_risk(hazard)[0])

        risks.append(risk)
        times.append(data["survival_time"].item())
        events.append(data["event"].item())
        pids.append(data.get("patient_id", f"P{i}"))

    return np.array(risks), np.array(times), np.array(events), pids

@torch.no_grad()
def _get_patient_risks_deepsurv(model, dataset, device):

    risks, times, events = [], [], []
    for i in range(len(dataset)):
        data = dataset[i]
        if not data.get("has_survival", False):
            continue
        x = data["clinical_features"].unsqueeze(0).to(device)
        out = model(x)
        hazard = torch.sigmoid(out).cpu().numpy()
        risk = float(hazard_to_risk(hazard)[0])
        risks.append(risk)
        times.append(data["survival_time"].item())
        events.append(data["event"].item())
    return np.array(risks), np.array(times), np.array(events)

@torch.no_grad()
def _get_patient_risks_graph_baseline(model, dataset, device, use_concepts, use_clinical):

    from plan3a.data.hypergraph import build_patient_hypergraph
    risks, times, events = [], [], []
    for i in range(len(dataset)):
        data = dataset[i]
        if not data.get("has_survival", False):
            continue
        nf = data["node_features"].to(device)
        if nf.dim() == 4:
            nf = nf.view(nf.shape[0], -1)
        hg = data.get("hyperedge_index")
        if hg is None:
            data = build_patient_hypergraph(data)
            hg = data["hyperedge_index"]
        hg = hg.to(device)
        nn_ = data["num_nodes"]
        ne = data.get("num_hyperedges", hg[1].max().item() + 1 if hg.shape[1] > 0 else 0)
        ct = data["concepts"].to(device) if use_concepts else None
        clin = data["clinical_features"].to(device) if use_clinical else None
        out = model(nf, hg, nn_, ne, concept_targets=ct, clinical_features=clin)
        hazard = torch.sigmoid(out["hazard_logits"]).cpu().numpy()
        risk = float(hazard_to_risk(hazard)[0])
        risks.append(risk)
        times.append(data["survival_time"].item())
        events.append(data["event"].item())
    return np.array(risks), np.array(times), np.array(events)

def _to_structured(times, events):

    return np.array(
        [(bool(e), t) for e, t in zip(events, times)],
        dtype=[("event", bool), ("time", float)],
    )

def compute_td_auc(risks, times, events, train_times, train_events, time_points=None):

    y_train = _to_structured(train_times, train_events)
    y_test = _to_structured(times, events)

    if time_points is None:

        time_points = [182, 365, 548]

    train_event_times = train_times[train_events.astype(bool)]
    test_event_times = times[events.astype(bool)]

    if len(train_event_times) == 0 or len(test_event_times) == 0:
        return {}

    tmin = max(train_event_times.min(), test_event_times.min()) + 1
    tmax = min(train_event_times.max(), test_event_times.max()) - 1
    valid_tp = [t for t in time_points if tmin < t < tmax]

    if not valid_tp:
        return {}

    try:
        aucs, mean_auc = cumulative_dynamic_auc(y_train, y_test, risks, valid_tp)
        result = {"mean_td_auc": float(mean_auc)}
        labels = {182: "6mo", 365: "12mo", 548: "18mo"}
        for tp, auc in zip(valid_tp, aucs):
            label = labels.get(tp, f"{tp}d")
            result[f"td_auc_{label}"] = float(auc)
        return result
    except Exception as e:
        print(f"    td-AUC failed: {e}")
        return {}

def compute_ibs(risks, times, events, train_times, train_events):

    y_train = _to_structured(train_times, train_events)
    y_test = _to_structured(times, events)

    train_event_times = train_times[train_events.astype(bool)]
    test_event_times = times[events.astype(bool)]

    if len(train_event_times) == 0 or len(test_event_times) == 0:
        return {}

    tmin = max(train_event_times.min(), test_event_times.min()) + 1
    tmax = min(train_event_times.max(), test_event_times.max()) - 1

    if tmin >= tmax:
        return {}

    eval_times = np.linspace(tmin, tmax, 50)

    max_t = max(times.max(), train_times.max())
    surv_probs = np.column_stack([
        np.exp(-risks * t / max_t) for t in eval_times
    ])

    try:
        ibs = integrated_brier_score(y_train, y_test, surv_probs, eval_times)
        return {"ibs": float(ibs)}
    except Exception as e:
        print(f"    IBS failed: {e}")
        return {}

def compute_km_stratification(risks, times, events):

    t33, t66 = np.percentile(risks, [33.3, 66.7])
    groups = np.where(risks <= t33, 0, np.where(risks <= t66, 1, 2))
    group_labels = ["Low", "Medium", "High"]

    km_results = {}
    for g, label in enumerate(group_labels):
        mask = groups == g
        if mask.sum() == 0:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(times[mask], events[mask], label=label)
        median = float(kmf.median_survival_time_)
        km_results[label] = {
            "n": int(mask.sum()),
            "median_survival": median if not np.isinf(median) else None,
            "events": int(events[mask].sum()),
        }

    high = groups == 2
    low = groups == 0
    if high.sum() > 0 and low.sum() > 0:
        lr = logrank_test(times[high], times[low], events[high], events[low])
        km_results["logrank_high_vs_low"] = {
            "test_statistic": float(lr.test_statistic),
            "p_value": float(lr.p_value),
        }

    return km_results

def evaluate_model_all_folds(
    model_name, checkpoints_dir, processed_dir, device,
    is_baseline=False, baseline_type=None,
):

    full_ds = Plan3aDataset(processed_dir, build_hypergraph=True)
    patient_ids = full_ds.patient_ids
    splits = get_kfold_splits(patient_ids, NUM_FOLDS)

    fold_metrics = []

    for fold_idx in range(NUM_FOLDS):
        split = splits[fold_idx]
        print(f"  Fold {fold_idx + 1}/{NUM_FOLDS}...")

        use_hg = True
        if not is_baseline and model_name in EXPERIMENTS:
            use_hg = EXPERIMENTS[model_name].get("use_hypergraph", True)

        train_ds = HypergraphDatasetWrapper(processed_dir, split["train"], use_hg)
        val_ds = HypergraphDatasetWrapper(processed_dir, split["val"], use_hg)

        train_t, train_e = [], []
        for i in range(len(train_ds)):
            d = train_ds[i]
            if d.get("has_survival", False):
                train_t.append(d["survival_time"].item())
                train_e.append(d["event"].item())
        train_t, train_e = np.array(train_t), np.array(train_e)

        try:
            if is_baseline:
                model = _load_baseline_model(baseline_type, fold_idx, checkpoints_dir, device)
                if model is None:
                    print(f"    Skipping fold {fold_idx} (no checkpoint)")
                    continue
                if baseline_type == "deepsurv":
                    risks, val_t, val_e = _get_patient_risks_deepsurv(model, val_ds, device)
                elif baseline_type == "hypercbm":
                    risks, val_t, val_e = _get_patient_risks_graph_baseline(
                        model, val_ds, device, use_concepts=True, use_clinical=False)
                elif baseline_type == "mrepath":
                    risks, val_t, val_e = _get_patient_risks_graph_baseline(
                        model, val_ds, device, use_concepts=False, use_clinical=True)
            else:
                model = _load_model(model_name, fold_idx, checkpoints_dir, device)
                risks, val_t, val_e, _ = _get_patient_risks_main(model, val_ds, device)
        except FileNotFoundError as e:
            print(f"    {e}")
            continue

        if len(risks) == 0:
            continue

        ci = concordance_index(risks, val_t, val_e)
        td = compute_td_auc(risks, val_t, val_e, train_t, train_e)
        ibs = compute_ibs(risks, val_t, val_e, train_t, train_e)
        km = compute_km_stratification(risks, val_t, val_e)

        fold_result = {
            "fold": fold_idx,
            "c_index": ci,
            **td,
            **ibs,
            "km_stratification": km,
        }
        fold_metrics.append(fold_result)

        print(f"    CI={ci:.4f}  td-AUC={td.get('mean_td_auc', 'N/A')}  "
              f"IBS={ibs.get('ibs', 'N/A')}")

        del model
        torch.cuda.empty_cache() if device != "cpu" else None

    return fold_metrics

def run_statistical_tests(results_dir):

    models = {
        "DeepSurv": "deepsurv", "HyperCBM": "hypercbm",
        "MRePath": "mrepath", "E7": "e7",
    }

    with open(os.path.join(results_dir, "ablation_results_e6.json")) as f:
        d = json.load(f)
    if isinstance(d, list): d = d[0]
    e6_folds = np.array([r["best_c_index"] for r in d["fold_results"]])

    results = {}
    for label, fname in models.items():
        path = os.path.join(results_dir, f"ablation_results_{fname}.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            d = json.load(f)
        if isinstance(d, list): d = d[0]
        other = np.array([r["best_c_index"] for r in d["fold_results"]])

        t_stat, t_p = ttest_rel(e6_folds, other)
        try:
            w_stat, w_p = wilcoxon(e6_folds, other)
        except:
            w_stat, w_p = float("nan"), float("nan")

        results[label] = {
            "mean_diff": float(np.mean(e6_folds - other)),
            "t_test_p": float(t_p),
            "wilcoxon_p": float(w_p),
        }
        print(f"  E6 vs {label:10s}: diff={np.mean(e6_folds-other):+.4f}  "
              f"t-test p={t_p:.4f}  Wilcoxon p={w_p:.4f}")

    return results

if __name__ == "__main__":
    checkpoints_dir = '/mnt/Stuff/arche/arche-brain-tumor-gnn/plan3a/checkpoints'
    processed_dir = None
    device = 'cuda'
    models = ["E6", "E7", "deepsurv", "mrepath", "hypercbm"]
    stat_only = None

    print("=" * 70)
    print("  EXTENDED SURVIVAL METRICS")
    print("=" * 70)

    print("\n--- Statistical Significance (E6 vs baselines) ---")
    sig_results = run_statistical_tests(checkpoints_dir)

    if stat_only:
        out_path = os.path.join(checkpoints_dir, "statistical_tests.json")
        with open(out_path, "w") as f:
            json.dump(sig_results, f, indent=2)
        print(f"\nSaved to {out_path}")
        sys.exit(0)

    all_results = {"statistical_tests": sig_results}
    baselines = {"deepsurv": "deepsurv", "hypercbm": "hypercbm", "mrepath": "mrepath"}

    for model_name in models:
        print(f"\n--- {model_name} ---")
        is_baseline = model_name.lower() in baselines
        baseline_type = baselines.get(model_name.lower())

        fold_metrics = evaluate_model_all_folds(
            model_name, checkpoints_dir, processed_dir,
            device, is_baseline=is_baseline, baseline_type=baseline_type,
        )

        if fold_metrics:

            agg = {
                "c_index": np.mean([f["c_index"] for f in fold_metrics]),
                "c_index_std": np.std([f["c_index"] for f in fold_metrics]),
            }
            for key in ["mean_td_auc", "td_auc_6mo", "td_auc_12mo", "td_auc_18mo", "ibs"]:
                vals = [f[key] for f in fold_metrics if key in f]
                if vals:
                    agg[key] = float(np.mean(vals))
                    agg[f"{key}_std"] = float(np.std(vals))

            all_results[model_name] = {"aggregate": agg, "folds": fold_metrics}

            print(f"\n  Summary: CI={agg['c_index']:.4f}  "
                  f"td-AUC={agg.get('mean_td_auc', 'N/A')}  "
                  f"IBS={agg.get('ibs', 'N/A')}")

    out_path = os.path.join(checkpoints_dir, "extended_metrics.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n{'='*70}")
    print(f"  Results saved to {out_path}")
    print(f"{'='*70}")
