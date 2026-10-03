import os
from pathlib import Path

PROJECT_ROOT = '/mnt/Stuff/arche/arche-brain-tumor-gnn'
DATA_ROOT = PROJECT_ROOT + "/upenn-filtered"
CLINICAL_CSV = DATA_ROOT + "/clinical_info.csv"
PROCESSED_DIR = PROJECT_ROOT + "/processed"

CORE_MODALITIES = ["T1-pre", "T1-post", "T2", "FLAIR"]

ADVANCED_MODALITIES = ["DTI", "Perfusion"]
ALL_MODALITIES = CORE_MODALITIES + ADVANCED_MODALITIES

PATCH_SIZE = 16
SLICE_STRIDE = 2
TARGET_SLICE_SIZE = (192, 192)
MIN_PATCH_INTENSITY = 0.02

NUM_CONCEPTS = 8

CLINICAL_CATEGORICAL = {
    "Gender": ["M", "F"],
    "IDH1": ["Wildtype", "Mutated"],
    "MGMT": ["Methylated", "Unmethylated"],
    "GTR_over90percent": ["Y", "N"],
}
CLINICAL_CONTINUOUS = ["Age_at_scan_years"]

CLINICAL_SPARSE = {
    "KPS": "continuous",
    "PsP_TP_score": "ordinal",
}

SURVIVAL_TIME_COL = "Survival_from_surgery_days_UPDATED"
SURVIVAL_STATUS_COL = "Survival_Status"

SURVIVAL_STATUS_MAP = {
    "Deceased": 1,
    "Deceased - uncertain date of death": 1,
    "Alive": 0,
    "Lost to Follow-up": 0,
}

TOPO_HYPEREDGE_RADIUS = 2.5
FEATURE_HYPEREDGE_K = 9
SHEAF_HGNN_LAYERS = 3
SHEAF_HGNN_DIM = 64

EMBED_DIM = 256
PATCH_ENCODER_CHANNELS = [32, 64]
NUM_CLINICAL_GROUPS = 5

BATCH_SIZE = 64
GRAD_ACCUM_STEPS = 4
LR = 1e-4
WEIGHT_DECAY = 1e-5
EPOCHS = 30
NUM_FOLDS = 5

EARLY_STOPPING_PATIENCE = 7
LR_WARMUP_EPOCHS = 3

USE_RANKING_LOSS = True
RANKING_LOSS_WEIGHT = 0.5

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EST_LAMBDA = 0.1
EST_WARMUP_EPOCHS = 3
EST_EVERY_N = 4
EST_TOP_K = 0.2

PREPROCESS_LIMIT = None
PREPROCESS_PATIENT_FILTER = None

TRAIN_LIMIT = None
TRAIN_FOLD = None

RUN_EXPERIMENT = "E7"
RUN_LIMIT = None
RUN_AUDIT = True

CHECKPOINT_EVERY = 1
RESUME_TRAINING = True

LOG_BACKEND = "tensorboard"
WANDB_PROJECT = "gbm-gnn-ablation"
WANDB_ENTITY = None
TENSORBOARD_DIR = None

RESULTS_JSON = None
REPORT_OUTPUT = None

CHECKPOINTS_DIR = PROJECT_ROOT + "/checkpoints"
