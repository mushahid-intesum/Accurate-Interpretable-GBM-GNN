import os
import warnings
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pydicom

warnings.filterwarnings("ignore", category=UserWarning, module="pydicom")

def _sort_key_for_dcm(filename: str) -> Tuple:

    stem = filename.replace(".dcm", "")
    parts = stem.split("-")
    return tuple(int(p) for p in parts)

def load_dicom_volume(dicom_dir: str) -> np.ndarray:

    dcm_files = sorted(
        [f for f in os.listdir(dicom_dir) if f.endswith(".dcm")],
        key=_sort_key_for_dcm,
    )
    if not dcm_files:
        raise FileNotFoundError(f"No DICOM files in {dicom_dir}")

    slices = []
    for f in dcm_files:
        ds = pydicom.dcmread(os.path.join(dicom_dir, f), force=True)
        if hasattr(ds, "pixel_array"):
            slices.append(ds.pixel_array.astype(np.float32))

    if not slices:
        raise ValueError(f"No readable pixel data in {dicom_dir}")

    return np.stack(slices, axis=0)

def load_structural_modality(patient_dir: str, modality: str) -> Tuple[np.ndarray, bool]:

    mod_dir = os.path.join(patient_dir, modality)
    if not os.path.isdir(mod_dir) or len(os.listdir(mod_dir)) == 0:
        return np.zeros((1, 1, 1), dtype=np.float32), False

    try:
        vol = load_dicom_volume(mod_dir)
        return vol, True
    except (FileNotFoundError, ValueError) as e:
        print(f"  Warning: {modality} failed for {patient_dir}: {e}")
        return np.zeros((1, 1, 1), dtype=np.float32), False

def load_dti_scalar(patient_dir: str) -> Tuple[np.ndarray, bool]:

    dti_dir = os.path.join(patient_dir, "DTI")
    if not os.path.isdir(dti_dir) or len(os.listdir(dti_dir)) == 0:
        return np.zeros((1, 1, 1), dtype=np.float32), False

    try:
        vol = load_dicom_volume(dti_dir)

        scalar_map = vol.mean(axis=0, keepdims=True)
        return scalar_map, True
    except (FileNotFoundError, ValueError) as e:
        print(f"  Warning: DTI failed for {patient_dir}: {e}")
        return np.zeros((1, 1, 1), dtype=np.float32), False

def load_perfusion_scalar(patient_dir: str) -> Tuple[np.ndarray, bool]:

    perf_dir = os.path.join(patient_dir, "Perfusion")
    if not os.path.isdir(perf_dir) or len(os.listdir(perf_dir)) == 0:
        return np.zeros((1, 1, 1), dtype=np.float32), False

    try:
        dcm_files = sorted(
            [f for f in os.listdir(perf_dir) if f.endswith(".dcm")],
            key=_sort_key_for_dcm,
        )
        if not dcm_files:
            return np.zeros((1, 1, 1), dtype=np.float32), False

        slices_by_tp = {}
        for f in dcm_files:
            stem = f.replace(".dcm", "")
            parts = stem.split("-")
            tp = int(parts[0])
            if tp not in slices_by_tp:
                slices_by_tp[tp] = []
            ds = pydicom.dcmread(os.path.join(perf_dir, f), force=True)
            if hasattr(ds, "pixel_array"):
                slices_by_tp[tp].append(ds.pixel_array.astype(np.float32))

        if not slices_by_tp:
            return np.zeros((1, 1, 1), dtype=np.float32), False

        tp_means = []
        for tp in sorted(slices_by_tp.keys()):
            if slices_by_tp[tp]:
                tp_vol = np.stack(slices_by_tp[tp], axis=0)
                tp_means.append(tp_vol.mean(axis=0))

        if not tp_means:
            return np.zeros((1, 1, 1), dtype=np.float32), False

        tp_stack = np.stack(tp_means, axis=0)
        rcbv_proxy = tp_stack.mean(axis=0, keepdims=True)
        return rcbv_proxy, True

    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"  Warning: Perfusion failed for {patient_dir}: {e}")
        return np.zeros((1, 1, 1), dtype=np.float32), False

def load_all_modalities(patient_dir: str) -> Dict[str, Tuple[np.ndarray, bool]]:

    result = {}

    for mod in ["T1-pre", "T1-post", "T2", "FLAIR"]:
        result[mod] = load_structural_modality(patient_dir, mod)

    result["DTI"] = load_dti_scalar(patient_dir)
    result["Perfusion"] = load_perfusion_scalar(patient_dir)

    return result

def normalize_volume(vol: np.ndarray, clip_percentile: float = 99.5) -> np.ndarray:

    if vol.max() == vol.min():
        return np.zeros_like(vol)
    clip_val = np.percentile(vol, clip_percentile)
    vol = np.clip(vol, 0, clip_val)
    vol = vol / (clip_val + 1e-8)
    return vol
