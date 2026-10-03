import csv
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    CLINICAL_CSV, CLINICAL_CATEGORICAL, CLINICAL_CONTINUOUS,
    CLINICAL_SPARSE, SURVIVAL_TIME_COL, SURVIVAL_STATUS_COL,
    SURVIVAL_STATUS_MAP,
)

def parse_clinical_csv(csv_path: Optional[str] = None) -> Dict[str, Dict]:

    if csv_path is None:
        csv_path = str(CLINICAL_CSV)

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    continuous_values = {col: [] for col in CLINICAL_CONTINUOUS}
    sparse_values = {col: [] for col in CLINICAL_SPARSE}

    for row in rows:
        for col in CLINICAL_CONTINUOUS:
            val = row.get(col, "").strip()
            try:
                continuous_values[col].append(float(val))
            except (ValueError, TypeError):
                pass
        for col in CLINICAL_SPARSE:
            val = row.get(col, "").strip()
            try:
                sparse_values[col].append(float(val))
            except (ValueError, TypeError):
                pass

    medians = {}
    for col in CLINICAL_CONTINUOUS:
        vals = continuous_values[col]
        medians[col] = np.median(vals) if vals else 0.0
    for col in CLINICAL_SPARSE:
        vals = sparse_values[col]
        medians[col] = np.median(vals) if vals else 0.0

    stds = {}
    means = {}
    for col in list(CLINICAL_CONTINUOUS) + list(CLINICAL_SPARSE.keys()):
        vals = continuous_values.get(col, sparse_values.get(col, []))
        if vals:
            means[col] = np.mean(vals)
            stds[col] = np.std(vals) + 1e-8
        else:
            means[col] = 0.0
            stds[col] = 1.0

    result = {}

    for row in rows:
        raw_id = row["ID"].strip()

        patient_id = raw_id.rsplit("_", 1)[0] if "_" in raw_id else raw_id

        features = []
        feature_names = []

        for col, valid_values in CLINICAL_CATEGORICAL.items():
            val = row.get(col, "").strip()
            is_missing = val not in valid_values
            one_hot = [1.0 if val == v else 0.0 for v in valid_values]
            features.extend(one_hot)
            features.append(1.0 if is_missing else 0.0)
            for v in valid_values:
                feature_names.append(f"{col}_{v}")
            feature_names.append(f"{col}_missing")

        for col in CLINICAL_CONTINUOUS:
            val = row.get(col, "").strip()
            try:
                numeric = float(val)
            except (ValueError, TypeError):
                numeric = medians[col]
                features.append((numeric - means[col]) / stds[col])
                features.append(1.0)
            else:
                features.append((numeric - means[col]) / stds[col])
                features.append(0.0)
            feature_names.append(f"{col}_norm")
            feature_names.append(f"{col}_missing")

        for col, dtype in CLINICAL_SPARSE.items():
            val = row.get(col, "").strip()
            try:
                numeric = float(val)
            except (ValueError, TypeError):
                numeric = medians[col]
                features.append((numeric - means[col]) / stds[col])
                features.append(1.0)
            else:
                features.append((numeric - means[col]) / stds[col])
                features.append(0.0)
            feature_names.append(f"{col}_norm")
            feature_names.append(f"{col}_missing")

        surv_time_str = row.get(SURVIVAL_TIME_COL, "").strip()
        surv_status_str = row.get(SURVIVAL_STATUS_COL, "").strip()

        try:
            survival_time = float(surv_time_str)
        except (ValueError, TypeError):
            survival_time = None

        event = SURVIVAL_STATUS_MAP.get(surv_status_str, None)

        result[patient_id] = {
            "features": np.array(features, dtype=np.float32),
            "feature_names": feature_names,
            "survival_time": survival_time,
            "event": event,
            "raw": dict(row),
        }

    return result

def get_feature_dim() -> int:

    dim = 0
    for col, valid_values in CLINICAL_CATEGORICAL.items():
        dim += len(valid_values) + 1
    for col in CLINICAL_CONTINUOUS:
        dim += 2
    for col in CLINICAL_SPARSE:
        dim += 2
    return dim

def get_survival_labels(
    clinical_data: Dict[str, Dict],
    patient_ids: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    times = []
    events = []
    valid = []

    for pid in patient_ids:
        info = clinical_data.get(pid, {})
        t = info.get("survival_time")
        e = info.get("event")
        if t is not None and e is not None:
            times.append(t)
            events.append(e)
            valid.append(True)
        else:
            times.append(0.0)
            events.append(0)
            valid.append(False)

    return (
        np.array(times, dtype=np.float32),
        np.array(events, dtype=np.int64),
        np.array(valid, dtype=bool),
    )
