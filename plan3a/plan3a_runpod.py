# %% [markdown]
# # Plan 3a — Full Ablation on RunPod
# 
# **Hypergraph Concept Bottleneck GNN for GBM Survival Prediction**
# 
# Runs the complete E1–E7 ablation study on the full UPenn-GBM dataset (~630 patients).
# - **Data**: Pre-downloaded at `/workspace/data/`
# - **Training**: 30 epochs, 5-fold CV, all 7 experiments
# - **Output**: Checkpoints, results JSON, and RESULTS.md

# %% [markdown]
# ## 1. Environment Setup

# %%
import subprocess, sys, os

# Install all project dependencies from requirements.txt
REPO_DIR = "/workspace/arche"
if not os.path.exists(os.path.join(REPO_DIR, "plan3a")):
    os.system(f"git clone https://github.com/mushahid-intesum/arche-brain-tumor-gnn.git {REPO_DIR}")
else:
    print(f"Repo already at {REPO_DIR}")

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "-r", os.path.join(REPO_DIR, "requirements.txt")
])

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"VRAM: {props.total_mem / 1e9:.1f} GB")

# %% [markdown]
# ## 2. Clone Repo & Configure Paths

# %%
import os, sys
from pathlib import Path

REPO_DIR = "/workspace/arche"
sys.path.insert(0, REPO_DIR)

# Override config paths for RunPod
import plan3a.config as cfg

cfg.DATA_ROOT = Path("/workspace/data")
cfg.CLINICAL_CSV = cfg.DATA_ROOT / "clinical_info.csv"
cfg.PROCESSED_DIR = Path("/workspace/processed")
cfg.CHECKPOINTS_DIR = Path("/workspace/checkpoints")

# Full training config — optimized for big GPU
cfg.EPOCHS = 30
cfg.NUM_FOLDS = 5
cfg.PREPROCESS_LIMIT = None
cfg.RUN_LIMIT = None
cfg.RUN_AUDIT = True
cfg.GRAD_ACCUM_STEPS = 2  # less accumulation on bigger VRAM

os.makedirs(str(cfg.PROCESSED_DIR), exist_ok=True)
os.makedirs(str(cfg.CHECKPOINTS_DIR), exist_ok=True)

print(f"DATA_ROOT:      {cfg.DATA_ROOT}")
print(f"CLINICAL_CSV:   {cfg.CLINICAL_CSV}")
print(f"PROCESSED_DIR:  {cfg.PROCESSED_DIR}")
print(f"CHECKPOINTS_DIR:{cfg.CHECKPOINTS_DIR}")
print(f"EPOCHS:         {cfg.EPOCHS}")
print(f"DEVICE:         {cfg.DEVICE}")

# %% [markdown]
# ## 3. Data Discovery & Validation

# %%
data_root = str(cfg.DATA_ROOT)

# Find patients
patient_dirs = sorted([
    d for d in os.listdir(data_root)
    if d.startswith("UPENN-GBM") and os.path.isdir(os.path.join(data_root, d))
])
print(f"Found {len(patient_dirs)} patient directories")

# Check clinical CSV
csv_exists = os.path.exists(str(cfg.CLINICAL_CSV))
print(f"Clinical CSV: {'✓' if csv_exists else '✗'} ({cfg.CLINICAL_CSV})")

# Show sample patient structure
if patient_dirs:
    sample = os.path.join(data_root, patient_dirs[0])
    contents = os.listdir(sample)
    print(f"\nSample patient ({patient_dirs[0]}):")
    for c in sorted(contents):
        full = os.path.join(sample, c)
        if os.path.isdir(full):
            n_files = len(os.listdir(full))
            print(f"  {c}/ ({n_files} files)")
        else:
            print(f"  {c}")

# Verify expected modalities
expected = {"T1-pre", "T1-post", "T2", "FLAIR"}
found = set(contents) if patient_dirs else set()
missing = expected - found
if missing:
    print(f"\n⚠️  Missing expected modalities: {missing}")
    print("Check if the data uses different naming conventions.")
else:
    print(f"\n✓ All core modalities found")

# %% [markdown]
# ## 4. Preprocessing (DICOM → .pt tensors)
# 
# This is the longest step (~15-20h for 630 patients). It's CPU-bound (DICOM I/O).
# Results are saved incrementally — safe to interrupt and resume.

# %%
from plan3a.data.preprocess import run_preprocessing
import time

# Check how many are already preprocessed
existing = [f for f in os.listdir(str(cfg.PROCESSED_DIR)) 
            if f.endswith('.pt') and f.startswith('UPENN')]
print(f"Already preprocessed: {len(existing)}/{len(patient_dirs)}")

if len(existing) < len(patient_dirs):
    print("Starting preprocessing...")
    t0 = time.time()
    run_preprocessing(
        data_root=str(cfg.DATA_ROOT),
        output_dir=str(cfg.PROCESSED_DIR),
        limit=None,
        verbose=True,
    )
    elapsed = time.time() - t0
    print(f"\nPreprocessing completed in {elapsed/3600:.1f} hours")
else:
    print("All patients already preprocessed. Skipping.")

# %% [markdown]
# ## 5. Preprocessing Sanity Check

# %%
import torch, numpy as np

processed_dir = str(cfg.PROCESSED_DIR)
pt_files = sorted([f for f in os.listdir(processed_dir) 
                   if f.endswith('.pt') and f.startswith('UPENN')])
print(f"Total preprocessed patients: {len(pt_files)}")

# Spot-check 3 patients
for f in pt_files[:3]:
    d = torch.load(os.path.join(processed_dir, f), weights_only=False)
    print(f"  {d['patient_id']}: {d['num_patches']} patches, "
          f"concepts={d['concepts'].shape}, "
          f"clinical={d['clinical_features'].shape}, "
          f"survival={d['survival_time']:.0f}d event={d['event']}")

# Summary stats
patch_counts = []
surv_count = 0
for f in pt_files:
    d = torch.load(os.path.join(processed_dir, f), weights_only=False)
    patch_counts.append(d['num_patches'])
    if d['has_survival']:
        surv_count += 1

print(f"\nPatch counts: min={min(patch_counts)}, max={max(patch_counts)}, "
      f"mean={np.mean(patch_counts):.0f}")
print(f"With survival data: {surv_count}/{len(pt_files)}")

# %% [markdown]
# ## 6. Run Full Ablation (E1–E7)
# 
# Each experiment runs 5-fold CV × 30 epochs. Results are saved incrementally
# after each experiment, so partial runs are recoverable.
#
# Estimated time per experiment: ~4-5h on A100.

# %%
from plan3a.runner import run_experiment, EXPERIMENTS
import json, time

experiments_to_run = ['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7']
results_path = os.path.join(str(cfg.CHECKPOINTS_DIR), 'ablation_results_full.json')

# Load any existing results (for resume)
if os.path.exists(results_path):
    with open(results_path) as f:
        results = json.load(f)
    completed = {r['experiment'] for r in results}
    print(f"Resuming — already completed: {completed}")
else:
    results = []
    completed = set()

for exp_id in experiments_to_run:
    if exp_id in completed:
        print(f"Skipping {exp_id} (already completed)")
        continue
    
    if exp_id not in EXPERIMENTS:
        print(f"Skipping {exp_id} (not defined)")
        continue
    
    print(f"\n{'='*70}")
    print(f"Starting {exp_id}: {EXPERIMENTS[exp_id]['name']}")
    print(f"{'='*70}")
    
    t0 = time.time()
    result = run_experiment(
        exp_id=exp_id,
        processed_dir=str(cfg.PROCESSED_DIR),
        epochs=cfg.EPOCHS,
        limit=None,
        device=cfg.DEVICE,
        save_dir=str(cfg.CHECKPOINTS_DIR),
        run_audit=cfg.RUN_AUDIT,
    )
    elapsed = time.time() - t0
    
    result['wall_time_hours'] = elapsed / 3600
    results.append(result)
    completed.add(exp_id)
    
    # Save incrementally
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✓ {exp_id}: C-Index = {result['mean_c_index']:.4f} ± "
          f"{result['std_c_index']:.4f} ({elapsed/3600:.1f}h)")

print(f"\n{'='*70}")
print("ALL EXPERIMENTS COMPLETE")
print(f"{'='*70}")

# %% [markdown]
# ## 7. Results Summary

# %%
# Reload results (in case notebook was restarted)
results_path = os.path.join(str(cfg.CHECKPOINTS_DIR), 'ablation_results_full.json')
with open(results_path) as f:
    results = json.load(f)

print(f"{'Exp':<5} {'Configuration':<45} {'C-Index':>14} {'Time':>7}")
print(f"{'-'*5} {'-'*45} {'-'*14} {'-'*7}")
for r in sorted(results, key=lambda x: x['experiment']):
    ci = f"{r['mean_c_index']:.4f}±{r['std_c_index']:.4f}"
    hours = f"{r.get('wall_time_hours', 0):.1f}h"
    print(f"{r['experiment']:<5} {r['name'][:45]:<45} {ci:>14} {hours:>7}")

# Per-fold breakdown
print(f"\n{'Exp':<5}", end="")
for i in range(5):
    print(f" {'Fold '+str(i+1):>8}", end="")
print(f" {'Mean':>8} {'Std':>8}")
print("-" * 60)

for r in sorted(results, key=lambda x: x['experiment']):
    print(f"{r['experiment']:<5}", end="")
    for fr in r['fold_results']:
        print(f" {fr['best_c_index']:>8.4f}", end="")
    print(f" {r['mean_c_index']:>8.4f} {r['std_c_index']:>8.4f}")

# %% [markdown]
# ## 8. Faithfulness Audit Summary

# %%
print(f"{'Exp':<5} {'EST Rej':>10} {'Fid- Rej':>10} {'Suf Rej':>10} {'Overall':>10}")
print(f"{'-'*5} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
for r in sorted(results, key=lambda x: x['experiment']):
    if 'faithfulness' in r:
        f = r['faithfulness']
        print(f"{r['experiment']:<5} "
              f"{f['est_rejection']:>9.0%} "
              f"{f['fid_minus_rejection']:>9.0%} "
              f"{f['sufficiency_rejection']:>9.0%} "
              f"{f['overall_rejection']:>9.0%}")
    else:
        print(f"{r['experiment']:<5} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10}")

# %% [markdown]
# ## 9. Generate Report

# %%
from plan3a.explain.report import generate_report

report_path = '/workspace/RESULTS.md'
generate_report(
    results_path=results_path,
    output_path=report_path,
)
print(f"\nReport saved to {report_path}")

# %% [markdown]
# ## 10. Save Artifacts

# %%
import shutil

artifacts_dir = '/workspace/artifacts'
os.makedirs(artifacts_dir, exist_ok=True)

# Copy results JSON
shutil.copy2(results_path, artifacts_dir)
print(f"✓ ablation_results_full.json")

# Copy report
if os.path.exists(report_path):
    shutil.copy2(report_path, artifacts_dir)
    print(f"✓ RESULTS.md")

# Copy best checkpoints (model weights)
ckpt_dir = str(cfg.CHECKPOINTS_DIR)
for f in os.listdir(ckpt_dir):
    if f.endswith('.pt'):
        shutil.copy2(os.path.join(ckpt_dir, f), artifacts_dir)
        print(f"✓ {f}")

print(f"\nAll artifacts saved to {artifacts_dir}")
print(f"Total size: {sum(os.path.getsize(os.path.join(artifacts_dir, f)) for f in os.listdir(artifacts_dir)) / 1e6:.1f} MB")
