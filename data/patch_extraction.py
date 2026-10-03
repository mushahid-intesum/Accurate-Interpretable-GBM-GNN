import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.ndimage import zoom

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    PATCH_SIZE, SLICE_STRIDE, TARGET_SLICE_SIZE, MIN_PATCH_INTENSITY
)
from data.dicom_loader import normalize_volume

def resize_slice(slice_2d: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:

    if slice_2d.shape == target_size:
        return slice_2d
    zoom_factors = (target_size[0] / slice_2d.shape[0],
                    target_size[1] / slice_2d.shape[1])
    return zoom(slice_2d, zoom_factors, order=1)

def extract_patches_from_slice(
    slice_2d: np.ndarray,
    patch_size: int
) -> Tuple[np.ndarray, np.ndarray]:

    H, W = slice_2d.shape
    rows = H // patch_size
    cols = W // patch_size

    patches = []
    coords = []
    for r in range(rows):
        for c in range(cols):
            patch = slice_2d[
                r * patch_size : (r + 1) * patch_size,
                c * patch_size : (c + 1) * patch_size,
            ]
            patches.append(patch)
            coords.append([
                (r + 0.5) * patch_size,
                (c + 0.5) * patch_size,
            ])

    return np.array(patches), np.array(coords)

def _align_volume_to_reference(vol: np.ndarray, ref_shape: Tuple[int, int]) -> np.ndarray:

    aligned = []
    for s in range(vol.shape[0]):
        aligned.append(resize_slice(vol[s], ref_shape))
    return np.stack(aligned, axis=0)

def extract_patient_patches(
    modality_volumes: Dict[str, Tuple[np.ndarray, bool]],
    patch_size: int = PATCH_SIZE,
    slice_stride: int = SLICE_STRIDE,
    target_size: Tuple[int, int] = TARGET_SLICE_SIZE,
    min_intensity: float = MIN_PATCH_INTENSITY,
) -> Dict:

    t1_pre_vol, t1_pre_ok = modality_volumes["T1-pre"]
    if not t1_pre_ok:
        raise ValueError("T1-pre must be present (reference modality)")

    t1_pre_vol = normalize_volume(t1_pre_vol)
    num_slices = t1_pre_vol.shape[0]
    ref_shape = (t1_pre_vol.shape[1], t1_pre_vol.shape[2])

    volumes = {"T1-pre": t1_pre_vol}
    modality_mask = {"T1-pre": True}

    for mod in ["T1-post", "T2", "FLAIR"]:
        vol, ok = modality_volumes[mod]
        modality_mask[mod] = ok
        if ok:
            vol = normalize_volume(vol)

            vol = _match_slice_count(vol, num_slices)

            vol = _align_volume_to_reference(vol, ref_shape)
        else:
            vol = np.zeros((num_slices,) + ref_shape, dtype=np.float32)
        volumes[mod] = vol

    for mod in ["DTI", "Perfusion"]:
        vol, ok = modality_volumes[mod]
        modality_mask[mod] = ok
        if ok:
            vol = normalize_volume(vol)

            if vol.shape[0] == 1:
                vol = np.repeat(vol, num_slices, axis=0)
            else:
                vol = _match_slice_count(vol, num_slices)
            vol = _align_volume_to_reference(vol, ref_shape)
        else:
            vol = np.zeros((num_slices,) + ref_shape, dtype=np.float32)
        volumes[mod] = vol

    all_patches = []
    all_coords = []
    all_concepts = []

    selected_slices = list(range(0, num_slices, slice_stride))

    for s_idx in selected_slices:

        slices = {}
        for mod in volumes:
            slc = volumes[mod][s_idx]
            slices[mod] = resize_slice(slc, target_size)

        patches_ref, coords_2d = extract_patches_from_slice(slices["T1-pre"], patch_size)
        num_p = patches_ref.shape[0]

        for p_idx in range(num_p):

            core_mean = np.mean([
                slices[m][
                    int(coords_2d[p_idx, 0] - patch_size/2):int(coords_2d[p_idx, 0] + patch_size/2),
                    int(coords_2d[p_idx, 1] - patch_size/2):int(coords_2d[p_idx, 1] + patch_size/2),
                ].mean()
                for m in ["T1-pre", "T1-post", "T2", "FLAIR"]
            ])
            if core_mean < min_intensity:
                continue

            r_start = int(coords_2d[p_idx, 0] - patch_size / 2)
            c_start = int(coords_2d[p_idx, 1] - patch_size / 2)
            r_end = r_start + patch_size
            c_end = c_start + patch_size

            channel_patches = []
            for mod in ["T1-pre", "T1-post", "T2", "FLAIR", "DTI", "Perfusion"]:
                channel_patches.append(slices[mod][r_start:r_end, c_start:c_end])
            multi_patch = np.stack(channel_patches, axis=0)

            z_norm = s_idx / max(num_slices - 1, 1)
            x_norm = coords_2d[p_idx, 0] / target_size[0]
            y_norm = coords_2d[p_idx, 1] / target_size[1]

            concepts = compute_patch_concepts(
                multi_patch, modality_mask, x_norm, y_norm, z_norm
            )

            all_patches.append(multi_patch)
            all_coords.append([x_norm, y_norm, z_norm])
            all_concepts.append(concepts)

    if not all_patches:

        return {
            "patches": np.zeros((0, 6, patch_size, patch_size), dtype=np.float32),
            "coords": np.zeros((0, 3), dtype=np.float32),
            "concepts": np.zeros((0, 8), dtype=np.float32),
            "modality_mask": modality_mask,
            "num_patches": 0,
        }

    return {
        "patches": np.stack(all_patches, axis=0).astype(np.float32),
        "coords": np.array(all_coords, dtype=np.float32),
        "concepts": np.stack(all_concepts, axis=0).astype(np.float32),
        "modality_mask": modality_mask,
        "num_patches": len(all_patches),
    }

def compute_patch_concepts(
    multi_patch: np.ndarray,
    modality_mask: Dict[str, bool],
    x_norm: float,
    y_norm: float,
    z_norm: float,
) -> np.ndarray:

    t1_pre = multi_patch[0]
    t1_post = multi_patch[1]
    t2 = multi_patch[2]
    flair = multi_patch[3]
    dti = multi_patch[4]
    perf = multi_patch[5]

    eps = 1e-8

    t1_pre_mean = t1_pre.mean() + eps
    t1_post_mean = t1_post.mean() + eps
    if modality_mask.get("T1-post", False) and t1_pre_mean > 0.01:
        ratio = t1_post_mean / t1_pre_mean
        c1 = np.clip(np.log1p(ratio), 0.0, 5.0)
    else:
        c1 = 0.0

    flair_mean = flair.mean()
    flair_std = flair.std() + eps
    c2 = flair_mean / flair_std if modality_mask.get("FLAIR", False) else 0.0

    t2_mean = t2.mean()
    c3 = t2_mean * flair_mean if modality_mask.get("T2", False) else 0.0

    c4 = dti.mean() if modality_mask.get("DTI", False) else 0.0

    dti_mean = dti.mean() + eps
    dti_std = dti.std()
    c5 = dti_std / dti_mean if modality_mask.get("DTI", False) else 0.0

    modality_means = [t1_pre_mean, t1_post_mean, t2_mean, flair_mean]
    if modality_mask.get("DTI", False):
        modality_means.append(dti.mean())
    if modality_mask.get("Perfusion", False):
        modality_means.append(perf.mean())
    c6 = np.std(modality_means)

    c7 = 0.0

    c8 = z_norm

    return np.array([c1, c2, c3, c4, c5, c6, c7, c8], dtype=np.float32)

def _match_slice_count(vol: np.ndarray, target_count: int) -> np.ndarray:

    current = vol.shape[0]
    if current == target_count:
        return vol
    elif current > target_count:

        start = (current - target_count) // 2
        return vol[start : start + target_count]
    else:

        pad_before = (target_count - current) // 2
        pad_after = target_count - current - pad_before
        return np.pad(vol, ((pad_before, pad_after), (0, 0), (0, 0)), mode="constant")
