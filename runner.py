import os
import sys
import json
import time

from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plan3a.config import (
    PROCESSED_DIR, LR, WEIGHT_DECAY, EPOCHS, NUM_FOLDS, DEVICE,
    SHEAF_HGNN_DIM, SHEAF_HGNN_LAYERS, NUM_CONCEPTS, GRAD_ACCUM_STEPS,
    RUN_EXPERIMENT, RUN_LIMIT, RUN_AUDIT, CHECKPOINTS_DIR,
    EST_LAMBDA, EST_WARMUP_EPOCHS, EST_EVERY_N, EST_TOP_K,
    CHECKPOINT_EVERY, RESUME_TRAINING,
    LOG_BACKEND, WANDB_PROJECT, WANDB_ENTITY, TENSORBOARD_DIR,
    EARLY_STOPPING_PATIENCE, LR_WARMUP_EPOCHS,
    USE_RANKING_LOSS, RANKING_LOSS_WEIGHT,
)
from plan3a.data.dataset import Plan3aDataset, get_kfold_splits
from plan3a.data.hypergraph import build_patient_hypergraph
from plan3a.model.full_model import Plan3aModel, NLLSurvivalLoss, CoxRankingLoss
from plan3a.eval.task_metrics import (
    concordance_index, concept_metrics, hazard_to_risk, compute_time_bins,
)
from plan3a.eval.faithfulness import (
    FaithfulnessAuditor, compute_rejection_ratios, _filter_hyperedges,
)

def init_logger(exp_id, fold_idx, log_dir=None):

    backend = LOG_BACKEND
    if backend is None:
        return None

    run_name = f"{exp_id}_fold{fold_idx}"

    if backend == "tensorboard":
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            print("    [warn] tensorboard not installed, skipping logging")
            return None
        tb_dir = log_dir or os.path.join(str(CHECKPOINTS_DIR), "tb_logs")
        writer = SummaryWriter(log_dir=os.path.join(tb_dir, run_name))
        return {"backend": "tensorboard", "writer": writer, "run_name": run_name}

    elif backend == "wandb":
        try:
            import wandb
        except ImportError:
            print("    [warn] wandb not installed, skipping logging")
            return None
        wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            name=run_name,
            config={"exp_id": exp_id, "fold": fold_idx,
                    "lr": LR, "epochs": EPOCHS, "weight_decay": WEIGHT_DECAY},
            reinit=True,
        )
        return {"backend": "wandb", "run_name": run_name}

    return None

def log_metrics(logger, metrics, step, prefix=""):

    if logger is None:
        return
    tag = f"{prefix}/" if prefix else ""

    if logger["backend"] == "tensorboard":
        writer = logger["writer"]
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                writer.add_scalar(f"{tag}{k}", v, step)
        writer.flush()

    elif logger["backend"] == "wandb":
        import wandb
        log_dict = {f"{tag}{k}": v for k, v in metrics.items()
                    if isinstance(v, (int, float))}
        log_dict["epoch"] = step
        wandb.log(log_dict, step=step)

def close_logger(logger):

    if logger is None:
        return
    if logger["backend"] == "tensorboard":
        logger["writer"].close()
    elif logger["backend"] == "wandb":
        import wandb
        wandb.finish()

def save_checkpoint(save_dir, exp_id, fold_idx, epoch, model, optimizer,
                    scheduler, best_ci, best_ep, history):

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{exp_id}_fold{fold_idx}_ckpt.pt")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_ci": best_ci,
        "best_ep": best_ep,
        "history": history,
    }, path)
    return path

def load_checkpoint(save_dir, exp_id, fold_idx, model, optimizer, scheduler, device):

    path = os.path.join(save_dir, f"{exp_id}_fold{fold_idx}_ckpt.pt")
    if not os.path.exists(path):
        return 0, 0.0, 0, []

    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    start_epoch = ckpt["epoch"]
    best_ci = ckpt["best_ci"]
    best_ep = ckpt["best_ep"]
    history = ckpt["history"]
    print(f"    ↻ Resumed from checkpoint: epoch {start_epoch}, "
          f"best C-Index={best_ci:.4f} @ ep {best_ep}")
    return start_epoch, best_ci, best_ep, history

EXPERIMENTS = {
    "E1": {
        "name": "E1: Baseline GNN (no hypergraph, no concepts)",
        "description": "kNN graph + simple GNN — no concept bottleneck, no fusion",
        "model_kwargs": {
            "use_hecrl": False,
            "residual_bypass": True,
            "use_fusion": False,
            "use_tree": False,
        },
        "use_hypergraph": False,
    },
    "E2": {
        "name": "E2: Hypergraph only (no concepts, no fusion)",
        "description": "Sheaf HGNN without concept bottleneck or clinical fusion",
        "model_kwargs": {
            "use_hecrl": False,
            "residual_bypass": True,
            "use_fusion": False,
            "use_tree": False,
        },
        "use_hypergraph": True,
    },
    "E3": {
        "name": "E3: Hypergraph + Concept Bottleneck",
        "description": "Core contribution: sheaf HGNN with concept bottleneck (imaging only)",
        "model_kwargs": {
            "use_hecrl": True,
            "residual_bypass": False,
            "use_fusion": False,
            "use_tree": False,
        },
        "use_hypergraph": True,
    },
    "E4": {
        "name": "E4: E3 + Clinical Fusion (MRePath-style)",
        "description": "Full imaging+clinical multimodal model",
        "model_kwargs": {
            "use_hecrl": True,
            "residual_bypass": False,
            "use_fusion": True,
            "use_tree": False,
        },
        "use_hypergraph": True,
    },
    "E5": {
        "name": "E5: E4 + TIF Multi-Granular Tree",
        "description": "Full model with hierarchical coarsening for multi-scale explanations",
        "model_kwargs": {
            "use_hecrl": True,
            "residual_bypass": False,
            "use_fusion": True,
            "use_tree": True,
            "tree_levels": 3,
        },
        "use_hypergraph": True,
    },
    "E6": {
        "name": "E6: E4 + EST Regularizer",
        "description": "Full model with faithfulness regularization during training",
        "model_kwargs": {
            "use_hecrl": True,
            "residual_bypass": False,
            "use_fusion": True,
            "use_tree": False,
        },
        "use_hypergraph": True,
        "est_regularize": True,
    },
    "E7": {
        "name": "E7: Full Model (Fusion + Tree + EST)",
        "description": "All components: SheafHGNN + CBM + Clinical Fusion + TIF Tree + EST Regularizer",
        "model_kwargs": {
            "use_hecrl": True,
            "residual_bypass": False,
            "use_fusion": True,
            "use_tree": True,
            "tree_levels": 3,
        },
        "use_hypergraph": True,
        "est_regularize": True,
    },
}

def build_knn_graph(data, k=8):

    from scipy.spatial.distance import cdist

    coords = data["coords"].numpy() if isinstance(data["coords"], torch.Tensor) else data["coords"]
    N = coords.shape[0]
    if N == 0:
        data["hyperedge_index"] = torch.zeros(2, 0, dtype=torch.long)
        data["num_hyperedges"] = 0
        data["node_features"] = torch.zeros(0, 1536)
        return data

    dist = cdist(coords, coords)
    np.fill_diagonal(dist, np.inf)

    node_list = []
    edge_list = []
    edge_id = 0

    for i in range(N):
        neighbors = np.argsort(dist[i])[:k]
        for j in neighbors:
            node_list.extend([i, j])
            edge_list.extend([edge_id, edge_id])
            edge_id += 1

    data["hyperedge_index"] = torch.tensor([node_list, edge_list], dtype=torch.long)
    data["num_hyperedges"] = edge_id
    patches = data["patches"]
    data["node_features"] = patches.reshape(N, -1).float() if isinstance(patches, torch.Tensor) else torch.from_numpy(patches.reshape(N, -1)).float()
    return data

def prepare_patient_data(patient_pt, use_hypergraph=True):

    if use_hypergraph:
        return build_patient_hypergraph(patient_pt)
    else:

        data = {
            "patches": patient_pt["patches"],
            "coords": patient_pt["coords"],
            "concepts": patient_pt["concepts"],
            "patient_id": patient_pt["patient_id"],
            "clinical_features": patient_pt["clinical_features"],
            "survival_time": patient_pt["survival_time"],
            "event": patient_pt["event"],
            "has_survival": patient_pt["has_survival"],
            "modality_mask": patient_pt["modality_mask"],
            "num_nodes": patient_pt["num_patches"],
        }
        return build_knn_graph(data)

def compute_est_loss(model, node_features, hyperedge_index,
                     num_nodes, num_edges, clinical_features,
                     full_hazard_logits, top_k=EST_TOP_K):

    with torch.no_grad():
        expl_outputs = model(
            node_features=node_features,
            hyperedge_index=hyperedge_index,
            num_nodes=num_nodes, num_edges=num_edges,
            clinical_features=clinical_features,
        )
        concepts = expl_outputs["concepts"]
        importance = torch.norm(concepts, dim=-1)
        k = max(1, int(num_nodes * top_k))
        _, top_indices = torch.topk(importance, k)
        expl_mask = torch.zeros(num_nodes, dtype=torch.bool,
                                device=node_features.device)
        expl_mask[top_indices] = True

    filtered_he, n_he = _filter_hyperedges(
        hyperedge_index, expl_mask, num_edges
    )

    if n_he == 0:
        return torch.tensor(0.0, device=node_features.device, requires_grad=True)

    expl_out = model(
        node_features=node_features,
        hyperedge_index=filtered_he,
        num_nodes=num_nodes, num_edges=n_he,
        clinical_features=clinical_features,
    )

    p_full = torch.sigmoid(full_hazard_logits.detach())
    p_expl = torch.sigmoid(expl_out["hazard_logits"])
    est_loss = (p_full - p_expl).abs().mean()

    return est_loss

def train_epoch(model, dataset, optimizer, time_bins, device,
                exp_config, current_epoch=0, grad_accum=GRAD_ACCUM_STEPS):

    model.train()
    optimizer.zero_grad()

    total_loss = 0.0
    total_surv = 0.0
    total_conc = 0.0
    total_est = 0.0
    total_rank = 0.0
    n = 0
    n_est = 0

    use_est = (
        exp_config.get("est_regularize", False)
        and current_epoch >= EST_WARMUP_EPOCHS
    )

    ranking_loss_fn = CoxRankingLoss() if USE_RANKING_LOSS else None
    accum_risks = []
    accum_times = []
    accum_events = []

    for i in range(len(dataset)):
        data = dataset[i]
        if data["num_nodes"] == 0 or not data["has_survival"]:
            continue

        nf = data["node_features"].to(device)
        he = data["hyperedge_index"].to(device)
        ct = data["concepts"].to(device) if "concepts" in data else None
        cl = data["clinical_features"].to(device)
        st = data["survival_time"].to(device).unsqueeze(0)
        ev = data["event"].to(device).unsqueeze(0)
        tb = time_bins.to(device)

        outputs = model(
            node_features=nf, hyperedge_index=he,
            num_nodes=data["num_nodes"], num_edges=data["num_hyperedges"],
            concept_targets=ct, clinical_features=cl,
        )

        losses = model.compute_loss(outputs, st, ev, tb)
        loss = losses["total_loss"]

        if use_est and (n % EST_EVERY_N == 0):
            est_loss = compute_est_loss(
                model, nf, he,
                data["num_nodes"], data["num_hyperedges"],
                cl, outputs["hazard_logits"],
            )
            loss = loss + EST_LAMBDA * est_loss
            total_est += est_loss.item()
            n_est += 1

        if ranking_loss_fn is not None:

            accum_risks.append(torch.sigmoid(outputs["hazard_logits"]).sum(dim=-1).squeeze().detach())
            accum_times.append(data["survival_time"].to(device).detach())
            accum_events.append(data["event"].to(device).detach())

        loss = loss / grad_accum
        loss.backward()

        if (i + 1) % grad_accum == 0 or (i + 1) == len(dataset):

            if ranking_loss_fn is not None and len(accum_risks) >= 2:

                re_out = model(
                    node_features=nf, hyperedge_index=he,
                    num_nodes=data["num_nodes"], num_edges=data["num_hyperedges"],
                    concept_targets=ct, clinical_features=cl,
                )
                current_risk = torch.sigmoid(re_out["hazard_logits"]).sum(dim=-1).squeeze()

                buf_risks = torch.stack(accum_risks[:-1])
                buf_times = torch.stack(accum_times[:-1]).float()
                buf_events = torch.stack(accum_events[:-1]).float()

                current_time = accum_times[-1].float()
                current_event = accum_events[-1].float()
                rank_loss = torch.tensor(0.0, device=device)
                n_pairs = 0

                if current_event.item() == 1:

                    later = buf_times > current_time
                    if later.any():
                        diff = current_risk - buf_risks[later]
                        rank_loss = rank_loss + (-F.logsigmoid(diff)).sum()
                        n_pairs += later.sum().item()

                died_before = (buf_events == 1) & (buf_times < current_time)
                if died_before.any():
                    diff = buf_risks[died_before] - current_risk
                    rank_loss = rank_loss + (-F.logsigmoid(diff)).sum()
                    n_pairs += died_before.sum().item()

                if n_pairs > 0:
                    rank_loss = RANKING_LOSS_WEIGHT * rank_loss / n_pairs
                    rank_loss.backward()
                    total_rank += (rank_loss.item() / RANKING_LOSS_WEIGHT)

                accum_risks.clear()
                accum_times.clear()
                accum_events.clear()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += losses["total_loss"].item()
        total_surv += losses["survival_loss"].item()
        total_conc += losses["concept_loss"].item()
        n += 1

    n_accum_steps = max(n // grad_accum, 1)
    return {
        "loss": total_loss / max(n, 1),
        "survival_loss": total_surv / max(n, 1),
        "concept_loss": total_conc / max(n, 1),
        "est_loss": total_est / max(n_est, 1),
        "ranking_loss": total_rank / n_accum_steps,
        "n_samples": n,
        "n_est_samples": n_est,
    }

@torch.no_grad()
def evaluate_epoch(model, dataset, time_bins, device):

    model.eval()
    risks, times, events = [], [], []
    pred_c, true_c = [], []
    total_loss = 0.0
    n = 0

    for i in range(len(dataset)):
        data = dataset[i]
        if data["num_nodes"] == 0 or not data["has_survival"]:
            continue

        nf = data["node_features"].to(device)
        he = data["hyperedge_index"].to(device)
        ct = data["concepts"].to(device) if "concepts" in data else None
        cl = data["clinical_features"].to(device)
        st = data["survival_time"].to(device).unsqueeze(0)
        ev = data["event"].to(device).unsqueeze(0)
        tb = time_bins.to(device)

        outputs = model(
            node_features=nf, hyperedge_index=he,
            num_nodes=data["num_nodes"], num_edges=data["num_hyperedges"],
            concept_targets=ct, clinical_features=cl,
        )
        losses = model.compute_loss(outputs, st, ev, tb)
        total_loss += losses["total_loss"].item()

        hazard = outputs["hazard_logits"].cpu().numpy()
        risks.append(hazard_to_risk(hazard)[0])
        times.append(data["survival_time"].item())
        events.append(data["event"].item())

        pred_c.append(outputs["concepts"].cpu().numpy().mean(0))
        true_c.append(ct.cpu().numpy().mean(0) if ct is not None else np.zeros(NUM_CONCEPTS))
        n += 1

    if n < 2:
        return {"c_index": 0.5, "loss": 0.0, "n_samples": n, "mean_concept_corr": 0.0}

    c_idx = concordance_index(np.array(risks), np.array(times), np.array(events))
    c_names = ["c1:enhance", "c2:flair", "c3:t2", "c4:dti_md",
               "c5:dti_fa", "c6:heterog", "c7:boundary", "c8:spatial"]
    cm = concept_metrics(np.stack(pred_c), np.stack(true_c), c_names)
    mean_corr = np.mean([v["pearson_r"] for k, v in cm.items() if k != "c7:boundary"])

    return {
        "c_index": c_idx,
        "loss": total_loss / n,
        "n_samples": n,
        "mean_concept_corr": mean_corr,
        "concept_metrics": cm,
    }

class HypergraphDatasetWrapper(Plan3aDataset):

    def __init__(self, processed_dir, patient_ids, use_hypergraph=True):
        super().__init__(processed_dir, patient_ids, build_hypergraph=False)
        self.use_hypergraph = use_hypergraph

    def __getitem__(self, idx):
        filepath = os.path.join(self.processed_dir, self.files[idx])
        patient_pt = torch.load(filepath, weights_only=False)
        return prepare_patient_data(patient_pt, self.use_hypergraph)

def run_experiment(
    exp_id: str,
    processed_dir: str,
    epochs: int,
    limit: int = None,
    device: str = DEVICE,
    save_dir: str = None,
    run_audit: bool = True,
) -> dict:

    exp_config = EXPERIMENTS[exp_id]
    print(f"\n{'#'*70}")
    print(f"# {exp_config['name']}")
    print(f"# {exp_config['description']}")
    print(f"{'#'*70}")

    all_pids = sorted([
        f.replace(".pt", "") for f in os.listdir(processed_dir)
        if f.endswith(".pt") and f.startswith("UPENN")
    ])
    if limit:
        all_pids = all_pids[:limit]

    splits = get_kfold_splits(all_pids, NUM_FOLDS)
    use_hg = exp_config.get("use_hypergraph", True)

    fold_results = []
    all_audit_reports = []

    for fold_idx in range(NUM_FOLDS):
        split = splits[fold_idx]
        print(f"\n  Fold {fold_idx+1}/{NUM_FOLDS}: "
              f"train={len(split['train'])}, val={len(split['val'])}")

        train_ds = HypergraphDatasetWrapper(processed_dir, split["train"], use_hg)
        val_ds = HypergraphDatasetWrapper(processed_dir, split["val"], use_hg)

        t_times, t_events = [], []
        for i in range(len(train_ds)):
            d = train_ds[i]
            if d["has_survival"]:
                t_times.append(d["survival_time"].item())
                t_events.append(d["event"].item())
        time_bins = torch.from_numpy(
            compute_time_bins(np.array(t_times), np.array(t_events), 4)
        )

        model = Plan3aModel(
            patch_dim=1536,
            embed_dim=SHEAF_HGNN_DIM,
            num_layers=SHEAF_HGNN_LAYERS,
            num_concepts=NUM_CONCEPTS,
            clinical_dim=18,
            num_survival_bins=4,
            **exp_config["model_kwargs"],
        ).to(device)

        n_params = sum(p.numel() for p in model.parameters())
        optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        if LR_WARMUP_EPOCHS > 0:
            from torch.optim.lr_scheduler import LinearLR, SequentialLR
            warmup_scheduler = LinearLR(
                optimizer, start_factor=0.01, end_factor=1.0,
                total_iters=LR_WARMUP_EPOCHS,
            )
            cosine_scheduler = CosineAnnealingLR(
                optimizer, T_max=max(epochs - LR_WARMUP_EPOCHS, 1),
                eta_min=LR * 0.01,
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[LR_WARMUP_EPOCHS],
            )
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=LR * 0.01)

        best_ci = 0.0
        best_ep = 0
        no_improve_count = 0
        history = []
        start_epoch = 0

        if RESUME_TRAINING and save_dir:
            start_epoch, best_ci, best_ep, history = load_checkpoint(
                save_dir, exp_id, fold_idx, model, optimizer, scheduler, device
            )

        if start_epoch >= epochs:
            print(f"    ✓ Fold already complete ({start_epoch}/{epochs} epochs)")
            fold_results.append({
                "fold": fold_idx,
                "best_c_index": best_ci,
                "best_epoch": best_ep,
                "n_params": n_params,
                "history": history,
                "n_audit": 0,
            })
            continue

        tb_dir = TENSORBOARD_DIR or (os.path.join(save_dir, "tb_logs") if save_dir else None)
        logger = init_logger(exp_id, fold_idx, log_dir=tb_dir)

        if start_epoch > 0 and logger is not None:
            for h in history:
                log_metrics(logger, {
                    "train_loss": h["train_loss"],
                    "val_c_index": h["val_c_index"],
                    "concept_corr": h["concept_corr"],
                    "est_loss": h.get("est_loss", 0),
                }, step=h["epoch"])

        for ep in range(start_epoch, epochs):
            t0 = time.time()
            tm = train_epoch(
                model, train_ds, optimizer, time_bins, device,
                exp_config, current_epoch=ep,
            )
            scheduler.step()
            vm = evaluate_epoch(model, val_ds, time_bins, device)
            elapsed = time.time() - t0

            current_lr = optimizer.param_groups[0]["lr"]

            est_str = f" est={tm['est_loss']:.4f}" if tm["est_loss"] > 0 else ""
            rank_str = f" rank={tm['ranking_loss']:.4f}" if tm.get('ranking_loss', 0) > 0 else ""
            print(f"    Ep {ep+1:3d} | loss={tm['loss']:.4f} "
                  f"surv={tm['survival_loss']:.4f} conc={tm['concept_loss']:.4f}"
                  f"{est_str}{rank_str} "
                  f"| val C-Idx={vm['c_index']:.4f} r={vm['mean_concept_corr']:.3f} "
                  f"| lr={current_lr:.2e} | {elapsed:.1f}s")

            log_metrics(logger, {
                "train/loss": tm["loss"],
                "train/survival_loss": tm["survival_loss"],
                "train/concept_loss": tm["concept_loss"],
                "train/est_loss": tm["est_loss"],
                "train/ranking_loss": tm.get("ranking_loss", 0),
                "val/c_index": vm["c_index"],
                "val/loss": vm["loss"],
                "val/concept_corr": vm["mean_concept_corr"],
                "lr": current_lr,
                "epoch_time_s": elapsed,
            }, step=ep + 1)

            if vm["c_index"] > best_ci:
                best_ci = vm["c_index"]
                best_ep = ep + 1
                no_improve_count = 0
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                    torch.save(model.state_dict(),
                               os.path.join(save_dir, f"{exp_id}_fold{fold_idx}_best.pt"))
            else:
                no_improve_count += 1

            history.append({
                "epoch": ep + 1,
                "train_loss": tm["loss"],
                "val_c_index": vm["c_index"],
                "concept_corr": vm["mean_concept_corr"],
                "est_loss": tm["est_loss"],
                "ranking_loss": tm.get("ranking_loss", 0),
            })

            if save_dir and (ep + 1) % CHECKPOINT_EVERY == 0:
                save_checkpoint(
                    save_dir, exp_id, fold_idx, ep + 1,
                    model, optimizer, scheduler, best_ci, best_ep, history
                )

            if EARLY_STOPPING_PATIENCE and no_improve_count >= EARLY_STOPPING_PATIENCE:
                print(f"    ⏹ Early stopping: no improvement for "
                      f"{EARLY_STOPPING_PATIENCE} epochs (best={best_ci:.4f} @ ep {best_ep})")
                break

        close_logger(logger)

        fold_audits = []
        if run_audit and use_hg:
            auditor = FaithfulnessAuditor(model, device=device, est_samples=20,
                                          prediction_threshold=0.1)
            for i in range(min(len(val_ds), 5)):
                data = val_ds[i]
                if data["num_nodes"] > 0 and data["has_survival"]:
                    report = auditor.audit_patient(data, top_k_ratio=0.2)
                    fold_audits.append(report)
                    all_audit_reports.append(report)

        if save_dir:
            ckpt_path = os.path.join(save_dir, f"{exp_id}_fold{fold_idx}_ckpt.pt")
            if os.path.exists(ckpt_path):
                os.remove(ckpt_path)

        print(f"    Best: C-Index={best_ci:.4f} @ epoch {best_ep}")

        fold_results.append({
            "fold": fold_idx,
            "best_c_index": best_ci,
            "best_epoch": best_ep,
            "n_params": n_params,
            "history": history,
            "n_audit": len(fold_audits),
        })

    c_indices = [r["best_c_index"] for r in fold_results]
    result = {
        "experiment": exp_id,
        "name": exp_config["name"],
        "description": exp_config["description"],
        "timestamp": datetime.now().isoformat(),
        "n_patients": len(all_pids),
        "n_folds": NUM_FOLDS,
        "epochs": epochs,
        "n_params": fold_results[0]["n_params"],
        "mean_c_index": float(np.mean(c_indices)),
        "std_c_index": float(np.std(c_indices)),
        "fold_results": fold_results,
    }

    if all_audit_reports:
        ratios = compute_rejection_ratios(all_audit_reports)
        result["faithfulness"] = ratios

    return result

def main():
    processed_dir = str(PROCESSED_DIR)
    save_dir = str(CHECKPOINTS_DIR)

    if RUN_EXPERIMENT == "all":
        experiments = list(EXPERIMENTS.keys())
    else:
        experiments = [RUN_EXPERIMENT]

    all_results = []

    for exp_id in experiments:
        result = run_experiment(
            exp_id=exp_id,
            processed_dir=processed_dir,
            epochs=EPOCHS,
            limit=RUN_LIMIT,
            device=DEVICE,
            save_dir=save_dir,
            run_audit=RUN_AUDIT,
        )
        all_results.append(result)

    print(f"\n{'='*80}")
    print(f"ABLATION RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"{'Exp':<5} {'Configuration':<45} {'C-Index':>12} {'Params':>10}")
    print(f"{'-'*5} {'-'*45} {'-'*12} {'-'*10}")
    for r in all_results:
        ci = f"{r['mean_c_index']:.4f}±{r['std_c_index']:.4f}"
        print(f"{r['experiment']:<5} {r['name'][:45]:<45} {ci:>12} {r['n_params']:>10,}")

    if any("faithfulness" in r for r in all_results):
        print(f"\n{'Exp':<5} {'EST Rej':>10} {'Fid- Rej':>10} {'Suf Rej':>10} {'Overall':>10}")
        print(f"{'-'*5} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
        for r in all_results:
            if "faithfulness" in r:
                f = r["faithfulness"]
                print(f"{r['experiment']:<5} "
                      f"{f['est_rejection']:>9.0%} "
                      f"{f['fid_minus_rejection']:>9.0%} "
                      f"{f['sufficiency_rejection']:>9.0%} "
                      f"{f['overall_rejection']:>9.0%}")
    print(f"{'='*80}")

    results_path = os.path.join(save_dir, "ablation_results_e7.json")
    os.makedirs(save_dir, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

if __name__ == "__main__":
    main()
