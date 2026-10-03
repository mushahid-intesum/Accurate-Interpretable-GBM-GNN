import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from scipy.spatial.distance import cdist

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    TOPO_HYPEREDGE_RADIUS, FEATURE_HYPEREDGE_K,
    EMBED_DIM, PATCH_SIZE,
)

def build_topological_hyperedges(
    coords: np.ndarray,
    radius: float = TOPO_HYPEREDGE_RADIUS,
    min_size: int = 2,
    max_size: int = 12,
) -> List[List[int]]:

    patch_grid_scale = PATCH_SIZE / 192.0
    norm_radius = radius * patch_grid_scale

    dist_matrix = cdist(coords, coords, metric="euclidean")

    hyperedges = []
    seen = set()

    for i in range(len(coords)):

        neighbors = np.where(dist_matrix[i] <= norm_radius)[0].tolist()

        if len(neighbors) < min_size:
            continue
        if len(neighbors) > max_size:

            dists = dist_matrix[i, neighbors]
            sorted_idx = np.argsort(dists)[:max_size]
            neighbors = [neighbors[j] for j in sorted_idx]

        key = frozenset(neighbors)
        if key not in seen:
            seen.add(key)
            hyperedges.append(sorted(neighbors))

    return hyperedges

def build_feature_hyperedges(
    features: np.ndarray,
    k: int = FEATURE_HYPEREDGE_K,
    min_size: int = 2,
) -> List[List[int]]:

    N = features.shape[0]

    norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
    features_norm = features / norms

    sim_matrix = features_norm @ features_norm.T

    hyperedges = []
    seen = set()

    for i in range(N):

        sims = sim_matrix[i].copy()
        sims[i] = -1
        top_k_idx = np.argsort(sims)[-k:]
        neighbors = [i] + top_k_idx.tolist()

        if len(neighbors) < min_size:
            continue

        key = frozenset(neighbors)
        if key not in seen:
            seen.add(key)
            hyperedges.append(sorted(neighbors))

    return hyperedges

def hyperedges_to_incidence(
    hyperedges: List[List[int]],
    num_nodes: int,
) -> torch.Tensor:

    node_indices = []
    hedge_indices = []

    for e_idx, hedge in enumerate(hyperedges):
        for n_idx in hedge:
            node_indices.append(n_idx)
            hedge_indices.append(e_idx)

    return torch.tensor([node_indices, hedge_indices], dtype=torch.long)

def build_patient_hypergraph(
    patient_data: Dict,
    use_concepts_for_features: bool = True,
    topo_radius: float = TOPO_HYPEREDGE_RADIUS,
    feature_k: int = FEATURE_HYPEREDGE_K,
) -> Dict:

    patches = patient_data["patches"]
    coords = patient_data["coords"]
    concepts = patient_data["concepts"]
    N = patches.shape[0]

    if N == 0:
        return _empty_hypergraph(patient_data)

    coords_np = coords.numpy() if isinstance(coords, torch.Tensor) else coords
    concepts_np = concepts.numpy() if isinstance(concepts, torch.Tensor) else concepts

    topo_hedges = build_topological_hyperedges(coords_np, radius=topo_radius)
    num_topo = len(topo_hedges)

    if use_concepts_for_features:
        feat_vectors = concepts_np
    else:

        patches_np = patches.numpy() if isinstance(patches, torch.Tensor) else patches
        feat_vectors = patches_np.reshape(N, -1)

    feat_hedges = build_feature_hyperedges(feat_vectors, k=feature_k)
    num_feat = len(feat_hedges)

    combined_hedges = topo_hedges + feat_hedges

    hyperedge_index_topo = hyperedges_to_incidence(topo_hedges, N)
    hyperedge_index_feat = hyperedges_to_incidence(feat_hedges, N)

    if num_feat > 0 and hyperedge_index_feat.shape[1] > 0:
        feat_offset = hyperedge_index_feat.clone()
        feat_offset[1] += num_topo
        hyperedge_index = torch.cat([hyperedge_index_topo, feat_offset], dim=1)
    else:
        hyperedge_index = hyperedge_index_topo

    hyperedge_type = torch.cat([
        torch.zeros(num_topo, dtype=torch.long),
        torch.ones(num_feat, dtype=torch.long),
    ])

    patches_t = patches if isinstance(patches, torch.Tensor) else torch.from_numpy(patches)
    node_features = patches_t.reshape(N, -1).float()

    result = {
        "node_features": node_features,
        "coords": coords if isinstance(coords, torch.Tensor) else torch.from_numpy(coords),
        "concepts": concepts if isinstance(concepts, torch.Tensor) else torch.from_numpy(concepts),
        "hyperedge_index_topo": hyperedge_index_topo,
        "hyperedge_index_feat": hyperedge_index_feat,
        "hyperedge_index": hyperedge_index,
        "hyperedge_type": hyperedge_type,
        "num_nodes": N,
        "num_hyperedges_topo": num_topo,
        "num_hyperedges_feat": num_feat,
        "num_hyperedges": num_topo + num_feat,

        "patient_id": patient_data["patient_id"],
        "clinical_features": patient_data["clinical_features"],
        "survival_time": patient_data["survival_time"],
        "event": patient_data["event"],
        "has_survival": patient_data["has_survival"],
        "modality_mask": patient_data["modality_mask"],
    }

    return result

def _empty_hypergraph(patient_data: Dict) -> Dict:

    return {
        "node_features": torch.zeros(0, 1536),
        "coords": torch.zeros(0, 3),
        "concepts": torch.zeros(0, 8),
        "hyperedge_index_topo": torch.zeros(2, 0, dtype=torch.long),
        "hyperedge_index_feat": torch.zeros(2, 0, dtype=torch.long),
        "hyperedge_index": torch.zeros(2, 0, dtype=torch.long),
        "hyperedge_type": torch.zeros(0, dtype=torch.long),
        "num_nodes": 0,
        "num_hyperedges_topo": 0,
        "num_hyperedges_feat": 0,
        "num_hyperedges": 0,
        "patient_id": patient_data.get("patient_id", "unknown"),
        "clinical_features": patient_data.get("clinical_features", torch.zeros(18)),
        "survival_time": patient_data.get("survival_time", torch.tensor(0.0)),
        "event": patient_data.get("event", torch.tensor(0)),
        "has_survival": False,
        "modality_mask": patient_data.get("modality_mask", {}),
    }
