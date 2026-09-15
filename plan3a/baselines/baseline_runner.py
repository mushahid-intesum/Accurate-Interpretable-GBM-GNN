"""
Baseline Runner — shared 5-fold CV evaluation for all baseline models.

Supports two training modes:
  1. Full-batch (DeepSurv): all patients loaded as tabular features
  2. Per-patient (HyperCBM, MRePath): gradient accumulation over graphs

Usage:
    python -m plan3a.baselines.baseline_runner --model deepsurv
    python -m plan3a.baselines.baseline_runner --model hypercbm
    python -m plan3a.baselines.baseline_runner --model mrepath
"""
import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from plan3a.config import (
    PROCESSED_DIR, EPOCHS, NUM_FOLDS, DEVICE,
    GRAD_ACCUM_STEPS, CHECKPOINTS_DIR as _CHECKPOINTS_DIR,
    EARLY_STOPPING_PATIENCE,
)

# Use local checkpoints dir if config path is not writable
_baselines_dir = Path(__file__).resolve().parent.parent / "checkpoints"
if os.path.isdir(os.path.dirname(_CHECKPOINTS_DIR)):
    CHECKPOINTS_DIR = _CHECKPOINTS_DIR
else:
    CHECKPOINTS_DIR = str(_baselines_dir)
from plan3a.data.dataset import Plan3aDataset, get_kfold_splits
from plan3a.eval.task_metrics import concordance_index, compute_time_bins, hazard_to_risk
from plan3a.model.full_model import NLLSurvivalLoss

from plan3a.baselines.deepsurv import DeepSurvModel, CoxPHLoss, DeepSurvWithNLL


# ── DeepSurv Training ────────────────────────────────────────────────────


def _collect_tabular_data(dataset):
    """
    Extract clinical features and survival labels from dataset
    into batch tensors for full-batch training.

    Returns:
        features: (N, 18) clinical feature matrix
        survival_time: (N,) survival times
        event: (N,) event indicators (1=deceased, 0=censored)
        patient_ids: list of patient IDs
    """
    features = []
    times = []
    events = []
    pids = []

    for i in range(len(dataset)):
        data = dataset[i]
        if not data.get("has_survival", False):
            continue

        features.append(data["clinical_features"])
        times.append(data["survival_time"])
        events.append(data["event"])
        pids.append(data["patient_id"])

    return (
        torch.stack(features),
        torch.stack(times).float(),
        torch.stack(events).long(),
        pids,
    )


def train_deepsurv_fold(
    train_ds, val_ds, fold_idx,
    device=DEVICE, lr=1e-3, weight_decay=1e-4,
    epochs=200, patience=20,
    use_nll=False, num_bins=4,
):
    """
    Train DeepSurv on one fold.

    Args:
        use_nll: if True, use discrete NLL loss (same as Plan3a).
                 if False, use original Cox PH partial likelihood.
    """
    # Collect all data into tensors
    train_x, train_t, train_e, _ = _collect_tabular_data(train_ds)
    val_x, val_t, val_e, _ = _collect_tabular_data(val_ds)

    train_x = train_x.to(device)
    train_t = train_t.to(device)
    train_e = train_e.to(device)
    val_x = val_x.to(device)
    val_t = val_t.to(device)
    val_e = val_e.to(device)

    in_dim = train_x.shape[1]

    if use_nll:
        model = DeepSurvWithNLL(in_dim=in_dim, num_bins=num_bins).to(device)
        time_bins = compute_time_bins(
            train_t.cpu().numpy(), train_e.cpu().numpy(), num_bins=num_bins
        )
        time_bins_t = torch.tensor(time_bins, device=device).float()
        nll_loss_fn = NLLSurvivalLoss(num_bins=num_bins)
    else:
        model = DeepSurvModel(in_dim=in_dim).to(device)
        cox_loss_fn = CoxPHLoss()

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_ci = 0.0
    best_epoch = 0
    patience_counter = 0
    history = []

    for epoch in range(1, epochs + 1):
        # ── Train ────────────────────────────────────────────────
        model.train()
        output = model(train_x)

        if use_nll:
            loss = nll_loss_fn(output, train_t, train_e, time_bins_t)
        else:
            loss = cox_loss_fn(output, train_t, train_e)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # ── Validate ─────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            val_output = model(val_x)

            if use_nll:
                val_loss = nll_loss_fn(val_output, val_t, val_e, time_bins_t)
                val_risk_np = hazard_to_risk(torch.sigmoid(val_output).cpu().numpy())
                val_risk = torch.tensor(val_risk_np)
            else:
                val_loss = cox_loss_fn(val_output, val_t, val_e)
                val_risk = val_output.squeeze(-1)

            ci = concordance_index(
                val_risk.cpu().numpy(),
                val_t.cpu().numpy(),
                val_e.cpu().numpy(),
            )

        history.append({
            "epoch": epoch,
            "train_loss": loss.item(),
            "val_loss": val_loss.item(),
            "val_c_index": ci,
        })

        # ── Early stopping ───────────────────────────────────────
        if ci > best_ci:
            best_ci = ci
            best_epoch = epoch
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if epoch % 20 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d}: loss={loss.item():.4f} "
                  f"val_ci={ci:.4f} best={best_ci:.4f}@{best_epoch}")

        if patience_counter >= patience:
            print(f"    Early stopping at epoch {epoch} (patience={patience})")
            break

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Fold {fold_idx+1}: best_ci={best_ci:.4f} @ epoch {best_epoch} "
          f"({len(history)} epochs, {n_params:,} params)")

    return {
        "fold": fold_idx,
        "best_c_index": best_ci,
        "best_epoch": best_epoch,
        "history": history,
        "n_params": n_params,
    }


# ── Generic Graph-Model Training ─────────────────────────────────────────


def train_graph_model_fold(
    model, train_ds, val_ds, fold_idx,
    device=DEVICE, lr=1e-4, weight_decay=1e-5,
    epochs=30, patience=7, grad_accum=4,
    use_concepts=False, use_clinical=False,
    num_bins=4,
):
    """
    Train a graph-based baseline (HyperCBM or MRePath) on one fold.

    Follows the same per-patient gradient accumulation as Plan3a runner.
    """
    from plan3a.data.hypergraph import build_patient_hypergraph

    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # Compute time bins from training data
    train_times = []
    train_events = []
    for i in range(len(train_ds)):
        d = train_ds[i]
        if d.get("has_survival", False):
            train_times.append(d["survival_time"].item())
            train_events.append(d["event"].item())
    time_bins = compute_time_bins(
        np.array(train_times), np.array(train_events), num_bins=num_bins
    )
    time_bins_t = torch.tensor(time_bins, device=device).float()
    nll_loss_fn = NLLSurvivalLoss(num_bins=num_bins)

    best_ci = 0.0
    best_epoch = 0
    patience_counter = 0
    history = []

    n_params = sum(p.numel() for p in model.parameters())

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # ── Train ────────────────────────────────────────────────
        model.train()
        optimizer.zero_grad()
        epoch_loss = 0.0
        epoch_concept_loss = 0.0
        n_train = 0

        indices = np.random.permutation(len(train_ds))

        for step, idx in enumerate(indices):
            data = train_ds[int(idx)]
            if not data.get("has_survival", False):
                continue

            # Prepare inputs
            node_feats = data["node_features"].to(device)
            if node_feats.dim() == 4:
                # Raw patches (N, 6, 16, 16) -> flatten
                node_feats = node_feats.view(node_feats.shape[0], -1)

            hg = data.get("hyperedge_index")
            if hg is None:
                data = build_patient_hypergraph(data)
                hg = data["hyperedge_index"]
            hg = hg.to(device)
            num_nodes = data["num_nodes"]
            num_edges = data.get("num_hyperedges", hg[1].max().item() + 1 if hg.shape[1] > 0 else 0)

            surv_t = data["survival_time"].to(device).unsqueeze(0)
            event = data["event"].to(device).unsqueeze(0)

            concepts_target = data["concepts"].to(device) if use_concepts else None
            clinical = data["clinical_features"].to(device) if use_clinical else None

            # Forward pass
            outputs = model(
                node_features=node_feats,
                hyperedge_index=hg,
                num_nodes=num_nodes,
                num_edges=num_edges,
                concept_targets=concepts_target,
                clinical_features=clinical,
            )

            # Compute loss
            if hasattr(model, "compute_loss"):
                loss_dict = model.compute_loss(outputs, surv_t, event, time_bins_t)
                loss = loss_dict["total_loss"]
                epoch_concept_loss += loss_dict.get("concept_loss", torch.tensor(0.0)).item()
            else:
                loss = nll_loss_fn(outputs["hazard_logits"], surv_t, event, time_bins_t)

            loss = loss / grad_accum
            loss.backward()
            epoch_loss += loss.item() * grad_accum
            n_train += 1

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

        # Final gradient step
        if n_train % grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        scheduler.step()
        avg_loss = epoch_loss / max(n_train, 1)

        # ── Validate ─────────────────────────────────────────────
        model.eval()
        val_risks = []
        val_times = []
        val_events = []

        with torch.no_grad():
            for i in range(len(val_ds)):
                data = val_ds[i]
                if not data.get("has_survival", False):
                    continue

                node_feats = data["node_features"].to(device)
                if node_feats.dim() == 4:
                    node_feats = node_feats.view(node_feats.shape[0], -1)
                hg = data.get("hyperedge_index")
                if hg is None:
                    data = build_patient_hypergraph(data)
                    hg = data["hyperedge_index"]
                hg = hg.to(device)
                num_nodes = data["num_nodes"]
                num_edges = data.get("num_hyperedges", hg[1].max().item() + 1 if hg.shape[1] > 0 else 0)

                concepts_target = data["concepts"].to(device) if use_concepts else None
                clinical = data["clinical_features"].to(device) if use_clinical else None

                outputs = model(
                    node_features=node_feats,
                    hyperedge_index=hg,
                    num_nodes=num_nodes,
                    num_edges=num_edges,
                    concept_targets=concepts_target,
                    clinical_features=clinical,
                )

                hazard_np = torch.sigmoid(outputs["hazard_logits"]).cpu().numpy()
                risk = hazard_to_risk(hazard_np)
                val_risks.append(float(risk[0]))
                val_times.append(data["survival_time"].item())
                val_events.append(data["event"].item())

        ci = concordance_index(
            np.array(val_risks), np.array(val_times), np.array(val_events)
        )

        elapsed = time.time() - t0
        history.append({
            "epoch": epoch,
            "train_loss": avg_loss,
            "val_c_index": ci,
            "concept_corr": 0.0,
        })

        # ── Early stopping ───────────────────────────────────────
        if ci > best_ci:
            best_ci = ci
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d}: loss={avg_loss:.4f} "
                  f"val_ci={ci:.4f} best={best_ci:.4f}@{best_epoch} "
                  f"({elapsed:.0f}s)")

        if patience_counter >= patience:
            print(f"    Early stopping at epoch {epoch}")
            break

    print(f"  Fold {fold_idx+1}: best_ci={best_ci:.4f} @ epoch {best_epoch} "
          f"({len(history)} epochs, {n_params:,} params)")

    return {
        "fold": fold_idx,
        "best_c_index": best_ci,
        "best_epoch": best_epoch,
        "history": history,
        "n_params": n_params,
    }


# ── Main Runner ──────────────────────────────────────────────────────────


def run_deepsurv(
    processed_dir=None, n_folds=NUM_FOLDS,
    epochs=200, patience=20, use_nll=True,
):
    """Run DeepSurv baseline with 5-fold CV."""
    if processed_dir is None:
        processed_dir = str(PROCESSED_DIR)

    print("=" * 70)
    print("BASELINE: DeepSurv (Katzman 2018)")
    print(f"  Mode: {'NLL (discrete bins)' if use_nll else 'Cox PH (continuous)'}")
    print(f"  Input: clinical features only (18-dim)")
    print("=" * 70)

    # Get all patient IDs
    full_ds = Plan3aDataset(processed_dir, build_hypergraph=False)
    patient_ids = full_ds.patient_ids
    print(f"  Patients: {len(patient_ids)}")

    splits = get_kfold_splits(patient_ids, n_folds)
    fold_results = []

    for fold_idx, split in enumerate(splits):
        print(f"\n  Fold {fold_idx + 1}/{n_folds} "
              f"(train={len(split['train'])}, val={len(split['val'])})")

        train_ds = Plan3aDataset(processed_dir, split["train"], build_hypergraph=False)
        val_ds = Plan3aDataset(processed_dir, split["val"], build_hypergraph=False)

        result = train_deepsurv_fold(
            train_ds, val_ds, fold_idx,
            epochs=epochs, patience=patience, use_nll=use_nll,
        )
        fold_results.append(result)

    # Aggregate results
    cis = [r["best_c_index"] for r in fold_results]
    mean_ci = np.mean(cis)
    std_ci = np.std(cis)

    print(f"\n{'=' * 70}")
    print(f"  DeepSurv Result: C-Index = {mean_ci:.4f} ± {std_ci:.4f}")
    print(f"{'=' * 70}")

    results = {
        "experiment": "DeepSurv",
        "name": "DeepSurv (Katzman 2018)",
        "description": "Cox PH neural network on clinical features only",
        "timestamp": datetime.now().isoformat(),
        "n_patients": len(patient_ids),
        "n_folds": n_folds,
        "epochs": epochs,
        "n_params": fold_results[0]["n_params"],
        "mean_c_index": mean_ci,
        "std_c_index": std_ci,
        "fold_results": fold_results,
    }

    # Save results
    out_path = os.path.join(CHECKPOINTS_DIR, "ablation_results_deepsurv.json")
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved to {out_path}")

    return results


def run_graph_baseline(
    model_name, model_factory,
    use_concepts=False, use_clinical=False,
    processed_dir=None, n_folds=NUM_FOLDS,
    epochs=30, patience=7, lr=1e-4,
):
    """Run a graph-based baseline with 5-fold CV."""
    if processed_dir is None:
        processed_dir = str(PROCESSED_DIR)

    print("=" * 70)
    print(f"BASELINE: {model_name}")
    print(f"  Concepts: {use_concepts}, Clinical fusion: {use_clinical}")
    print("=" * 70)

    full_ds = Plan3aDataset(processed_dir, build_hypergraph=True)
    patient_ids = full_ds.patient_ids
    print(f"  Patients: {len(patient_ids)}")

    splits = get_kfold_splits(patient_ids, n_folds)
    fold_results = []

    for fold_idx, split in enumerate(splits):
        print(f"\n  Fold {fold_idx + 1}/{n_folds} "
              f"(train={len(split['train'])}, val={len(split['val'])})")

        train_ds = Plan3aDataset(processed_dir, split["train"], build_hypergraph=True)
        val_ds = Plan3aDataset(processed_dir, split["val"], build_hypergraph=True)

        model = model_factory()

        result = train_graph_model_fold(
            model, train_ds, val_ds, fold_idx,
            epochs=epochs, patience=patience, lr=lr,
            use_concepts=use_concepts, use_clinical=use_clinical,
        )
        fold_results.append(result)

    cis = [r["best_c_index"] for r in fold_results]
    mean_ci = np.mean(cis)
    std_ci = np.std(cis)

    print(f"\n{'=' * 70}")
    print(f"  {model_name} Result: C-Index = {mean_ci:.4f} ± {std_ci:.4f}")
    print(f"{'=' * 70}")

    results = {
        "experiment": model_name,
        "name": model_name,
        "description": f"Standalone {model_name} baseline",
        "timestamp": datetime.now().isoformat(),
        "n_patients": len(patient_ids),
        "n_folds": n_folds,
        "epochs": epochs,
        "n_params": fold_results[0]["n_params"],
        "mean_c_index": mean_ci,
        "std_c_index": std_ci,
        "fold_results": fold_results,
    }

    safe_name = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    out_path = os.path.join(CHECKPOINTS_DIR, f"ablation_results_{safe_name}.json")
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved to {out_path}")

    return results


def run_hypercbm(processed_dir=None, n_folds=NUM_FOLDS, epochs=30, patience=7):
    """Run standalone HyperCBM baseline with 5-fold CV."""
    from plan3a.baselines.hypercbm import StandaloneHyperCBM

    return run_graph_baseline(
        model_name="HyperCBM",
        model_factory=lambda: StandaloneHyperCBM(),
        use_concepts=True,
        use_clinical=False,
        processed_dir=processed_dir,
        n_folds=n_folds,
        epochs=epochs,
        patience=patience,
    )


# ── CLI ──────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run baseline models")
    parser.add_argument(
        "--model", type=str, required=True,
        choices=["deepsurv", "hypercbm", "mrepath", "all"],
        help="Which baseline to run",
    )
    parser.add_argument("--processed-dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    if args.model in ("deepsurv", "all"):
        run_deepsurv(
            processed_dir=args.processed_dir,
            epochs=args.epochs or 200,
        )

    if args.model in ("hypercbm", "all"):
        from plan3a.baselines.hypercbm import StandaloneHyperCBM

        run_graph_baseline(
            model_name="HyperCBM",
            model_factory=lambda: StandaloneHyperCBM(),
            use_concepts=True,
            use_clinical=False,
            processed_dir=args.processed_dir,
            epochs=args.epochs or 30,
        )

    if args.model in ("mrepath", "all"):
        # Will be implemented in mrepath.py
        print("MRePath: not yet implemented")
