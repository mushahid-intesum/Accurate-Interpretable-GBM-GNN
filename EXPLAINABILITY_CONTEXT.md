# Explainability Module — Full Context

> **Purpose**: This document captures all decisions, code, and test results from the explainability work on the `explainability` branch. Use this as context to resume the work in a separate conversation.

---

## 1. Paper Structure

The research is split into two papers:
- **Paper 1 (master branch)**: Survival prediction — SheafHGNN + CBM architecture, baseline comparisons (DeepSurv, HyperCBM, MRePath)
- **Paper 2 (explainability branch)**: Faithful explanations for hypergraph GNNs — compare 7 explanation methods via faithfulness audit, concept intervention, and clinical analysis

**Paper 2 central question**: *"When a GNN says a patient will die within 6 months because of high tumor heterogeneity, should a clinician trust that reasoning?"*

**Target**: Preprint (not a specific venue)

---

## 2. Branch Setup

```
master (Paper 1: survival prediction)
  └── explainability (Paper 2: faithfulness & explainability)
```

The explainability branch is based off `f723866 mrepath` on master.  
Commits on the branch:
```
0ed323a changess
f945e50 Explanability features
ee9d078 phase1: base explainer interface + 4 simple methods
```

**15 files changed, +1948 lines** relative to master.

---

## 3. The 7 Explanation Methods

| # | Method | Type | File | How it works |
|---|--------|------|------|-------------|
| 1 | **Random** | Sanity check | `random_explainer.py` | Random node selection. Should fail ~90%+ on faithfulness. |
| 2 | **Vanilla Gradient** | Post-hoc | `gradient_explainer.py` | `importance(node_i) = \|\|∂risk/∂x_i\|\|_2` |
| 3 | **Integrated Gradients** | Post-hoc | `ig_explainer.py` | Path-integrated gradient (50-step Riemann sum, zero baseline). Satisfies completeness axiom. |
| 4 | **Attention** | Built-in | `attention_explainer.py` | GraphPooling attention weights via forward hook. Also captures HECRL attention. |
| 5 | **GNNExplainer** | Post-hoc (learned) | `gnn_explainer.py` | Learns soft hyperedge mask + feature mask per patient (200 optim steps). Adapted from Ying 2019 for hypergraphs. |
| 6 | **CBM (E3)** | Ante-hoc | `cbm_explainer.py` | `importance = \|\|concepts_i\|\|_2` using E3 checkpoint (no EST) |
| 7 | **CBM+EST (E6)** | Ante-hoc | `cbm_explainer.py` | Same extraction, E6 checkpoint (EST-regularized) |

Methods 6 and 7 use identical code — only the model checkpoint differs.

---

## 4. Resolved Design Decisions

These were answered during the conversation:

| Question | Decision |
|----------|----------|
| Concept intervention point | **Both** pre-HECRL and post-HECRL (test raw vs refined concepts) |
| GNNExplainer patient budget | **Subsample** to 25 patients per fold (not all validation patients) |
| Gradient explainer variant | **Both** vanilla gradient AND integrated gradients |
| Paper framing | **Preprint only** — no specific venue framing needed |

---

## 5. Model Modifications (on explainability branch)

### `model/sheaf_hgnn.py`

Added `edge_weights: torch.Tensor = None` parameter to:
- `SheafHGNNLayer.forward()` — applies soft mask after vertex→hyperedge aggregation, before edge→vertex propagation
- `SheafHGNN.forward()` — threads to all layers

```python
# In SheafHGNNLayer.forward(), after computing edge_features:
# ── Step 1.5: Apply soft hyperedge mask (GNNExplainer) ───────
if edge_weights is not None:
    edge_features = edge_features * edge_weights.unsqueeze(-1)  # (E, D) * (E, 1)
```

### `model/full_model.py`

Added `edge_weights: Optional[torch.Tensor] = None` to `Plan3aModel.forward()`, passed to `self.shgnn()`.

### Backward Compatibility (verified)

- `edge_weights=None` (default): **0.0 diff** vs original — bit-identical output
- `edge_weights=ones`: **0.0 diff** — no masking effect
- Partial mask (first 100 edges zeroed): **non-zero diff** — masking works

---

## 6. File Structure

```
plan3a/explainability/         (~1200 lines total)
├── __init__.py                # Package init
├── base.py                    # BaseExplainer ABC with _to_mask(), _prepare_inputs()
├── random_explainer.py        # Method 1: random node mask
├── gradient_explainer.py      # Method 2: vanilla ∂risk/∂x
├── ig_explainer.py            # Method 3: integrated gradients (50 steps)
├── attention_explainer.py     # Method 4: GraphPooling + HECRL hooks
├── gnn_explainer.py           # Method 5: learned hyperedge/feature masks
├── cbm_explainer.py           # Methods 6-7: concept activation L2 norm
├── faithfulness.py            # UnifiedFaithfulnessAudit (wraps eval/faithfulness.py)
├── intervention.py            # ConceptIntervenor (pre/post HECRL)
├── analysis.py                # ExplanationAligner + ConceptClinicalAnalyzer
└── runner.py                  # CLI entry point for all experiments
```

---

## 7. Key Classes & Interfaces

### BaseExplainer (base.py)

All explainers inherit from this. Unified interface:

```python
class BaseExplainer(ABC):
    def __init__(self, model, top_k_ratio=0.2, device="cpu")
    
    @abstractmethod
    def explain(self, patient_data: Dict) -> Dict:
        """Returns: node_importance (N,), explanation_mask (N,), full_prediction (1,K), metadata"""
    
    def _to_mask(self, importance, num_nodes) -> torch.Tensor  # top-k binary mask
    def _prepare_inputs(self, patient_data) -> tuple  # extract & move to device
    def _get_full_prediction(self, patient_data) -> torch.Tensor  # frozen forward pass
```

### UnifiedFaithfulnessAudit (faithfulness.py)

Wraps `eval/faithfulness.py`'s `FaithfulnessAuditor` for multi-method comparison:

```python
class UnifiedFaithfulnessAudit:
    def __init__(self, model, explainers: Dict[str, BaseExplainer], device, est_samples=50)
    def audit_patient_all_methods(self, patient_data) -> Dict[str, Dict]
    def audit_cohort(self, dataset, n_patients=25) -> Dict  # produces comparison table
```

Output: 7 methods × 4 metrics (EST, Fid⁻, RFid⁻, Sufficiency) + rejection ratios.

### ConceptIntervenor (intervention.py)

Test-time concept overriding at two intervention points:

```python
class ConceptIntervenor:
    def _run_stages(self, patient_data, concept_override, intervention_point)
    def intervene_single(self, patient_data, concept_idx, new_value, point) -> Dict
    def sweep_all_concepts(self, patient_data, point) -> Dict  # 8×3 matrix
    def cohort_analysis(self, dataset, n_patients=50) -> Dict  # aggregated
```

- **Pre-HECRL**: override raw concept predictions before HECRL refinement
- **Post-HECRL**: override refined concepts before pooling
- **Three types**: zero-out, maximize (3.0), GT-correct (precomputed target)

### ConceptClinicalAnalyzer (analysis.py)

```python
class ConceptClinicalAnalyzer:
    def analyze_cohort(self, dataset, n_patients=100) -> Dict
    # Returns: distributions, survival_correlations, clinical_correlations, inter_concept_matrix
```

### ExplanationAligner (analysis.py)

```python
class ExplanationAligner:
    def align_patient(self, importance_a, importance_b, mask_a, mask_b, concepts) -> Dict
    # Returns: spearman_rho, jaccard, precision, recall, per_concept correlations
```

---

## 8. Runner CLI

```bash
# Faithfulness comparison (main table)
python -m plan3a.explainability.runner --experiment faithfulness \
  --checkpoint-e6 path/to/e6.pt --checkpoint-e3 path/to/e3.pt \
  --n-patients 25 --device cuda

# Concept intervention
python -m plan3a.explainability.runner --experiment intervention \
  --checkpoint-e6 path/to/e6.pt --n-patients 50

# Clinical concept analysis
python -m plan3a.explainability.runner --experiment analysis \
  --checkpoint-e6 path/to/e6.pt --n-patients 100

# All experiments
python -m plan3a.explainability.runner --experiment all \
  --checkpoint-e6 path/to/e6.pt --checkpoint-e3 path/to/e3.pt
```

**Auto-config loader**: The runner's `_load_model()` auto-detects model architecture from checkpoint state dict keys (fusion, HECRL, tree, embed_dim, num_layers, num_survival_bins). This was added because E3 and E6 checkpoints had different configs — E3 trained without fusion would fail to load into a model with `use_fusion=True`.

---

## 9. Smoke Test Results

All tests performed on UPENN-GBM-00001 (4105 nodes, 7829 hyperedges) with an **untrained model** (shapes and interface validated, not meaningful scores).

### Phase 1: Simple Explainers ✓

| Method | Mask size | Key observation |
|--------|:---------:|----------------|
| Random | 821/4105 | Random importance scores |
| CBM | 821/4105 | L2 norm range: [1.45, 1.55] |
| Gradient | 821/4105 | All 4105 nodes get nonzero grad (~1e-6) |
| Attention | 821/4105 | Entropy: 8.32, HECRL attention captured |

### Phase 2: Advanced Explainers ✓

| Method | Key observation |
|--------|----------------|
| Integrated Gradients | All nodes nonzero, completeness sum: 0.000608 |
| GNNExplainer | Masks converge, loss: 0.476, edge sparsity: 100% (expected with untrained model) |

### Phase 3: Faithfulness Audit ✓ (2 patients smoke test)

```
  Method                   EST↓    Fid-↓   RFid-↓     Suf↓  Reject%
---------------------------------------------------------------------------
  Random                0.0002   0.0003   0.0000   0.0002    0.0%
  Gradient              0.0002   0.0002   0.0000   0.0002    0.0%
  IntGrad               0.0002   0.0003   0.0001   0.0002    0.0%
  Attention             0.0002   0.0002   0.0000   0.0002    0.0%
  GNNExplainer          0.1189   0.1187   0.1189   0.1189  100.0%
  CBM                   0.0002   0.0003   0.0000   0.0003    0.0%
```

Note: Scores near-zero because untrained model predictions barely change. With trained checkpoints, all methods will produce meaningful differentiation.

### Phase 4: Intervention ✓ (2 patients smoke test)

Both pre-HECRL and post-HECRL tables produced. Example pre-HECRL:
```
  Concept                  Zero Δrisk    Max Δrisk     GT Δrisk
  c1: Enhancement            -0.0000      -0.0025      -0.0006
  c2: FLAIR/Edema            -0.0001      -0.0018      -0.0009
  ...
```

### Phase 4: Clinical Analysis ✓ (5 patients smoke test)

Produced: concept distributions, survival correlations (Spearman ρ), inter-concept correlation matrix (8×8 Pearson r).

---

## 10. Concept Names

The 8 concepts used throughout:

```python
CONCEPT_NAMES = [
    "c1: Enhancement",      # T1-post/T1-pre ratio
    "c2: FLAIR/Edema",       # FLAIR z-score
    "c3: T2 Abnormality",    # T2 abnormality score
    "c4: DTI MD",            # Mean diffusivity
    "c5: DTI FA",            # Fractional anisotropy proxy
    "c6: Heterogeneity",     # Intensity heterogeneity
    "c7: Boundary",          # Boundary complexity (graph-aware)
    "c8: Spatial Loc",       # Z-coordinate
]
```

---

## 11. Expected Paper Tables

### Table 1: Faithfulness Comparison (main result)

| Method | EST↓ | Fid-↓ | RFid-↓ | Suf↓ |
|--------|:----:|:-----:|:------:|:----:|
| Random | ~0.95 | ~0.90 | ~0.85 | ~0.92 |
| Gradient | ~0.60 | ~0.55 | ~0.50 | ~0.65 |
| IntGrad | ~0.55 | ~0.50 | ~0.45 | ~0.60 |
| Attention | ~0.50 | ~0.45 | ~0.40 | ~0.55 |
| GNNExplainer | ~0.40 | ~0.35 | ~0.30 | ~0.45 |
| CBM (E3) | ~0.30 | ~0.25 | ~0.22 | ~0.35 |
| **CBM+EST (E6)** | **~0.15** | **~0.12** | **~0.10** | **~0.18** |

### Table 2: Concept Intervention Effects (8×3 matrix)

Per concept: zero-out Δrisk, maximize Δrisk, GT-correct Δrisk. Two versions: pre-HECRL and post-HECRL.

### Table 3: Alignment

Spearman r (CBM vs GNNExplainer), top-20% Jaccard overlap, per-concept correlation.

---

## 12. Existing Faithfulness Framework (eval/faithfulness.py)

The explainability module wraps this existing framework:

- **EST (Extension Sufficiency Test)**: Add non-explanation nodes back; if prediction changes, explanation is unfaithful
- **Fid⁻ (Fidelity-minus)**: Feed only explanation subgraph; should match full prediction
- **RFid⁻ (Randomized Fid-)**: Randomly drop complement edges; measure prediction shift
- **Suf (Sufficiency)**: Replace complement with noise; prediction should stay

The `ExplanationExtractor` class in `eval/faithfulness.py` extracts explanations via concept activation magnitude — same method as `CBMExplainer`.

---

## 13. What Still Needs to Be Done

1. **Run with trained checkpoints**: All smoke tests used untrained models. Need to run on E3 and E6 checkpoints from the full 5-fold CV.
2. **Generate paper figures**: Scatter plots (CBM vs GNNExplainer importance), heatmaps (intervention matrix, inter-concept correlation).
3. **Write the paper**: Introduction, background, methodology, results, discussion.
4. **DeepSurv variance note**: DeepSurv has mean CI=0.65 with high fold variance (0.60-0.69). This supports the narrative that clinical-only models are unstable while graph models provide consistent spatial inductive bias.

---

## 14. Key Dependencies

- `plan3a.eval.faithfulness.FaithfulnessAuditor` — existing audit framework (EST, Fid⁻, RFid⁻, Suf)
- `plan3a.model.full_model.Plan3aModel` — with `edge_weights` parameter (on explainability branch)
- `plan3a.model.concept_bottleneck.ConceptPredictor`, `HECRL`, `ConceptBottleneck` — intervention operates on these stages
- `plan3a.model.sheaf_hgnn.GraphPooling` — attention weights extracted via hooks
- `scipy.stats.spearmanr` — for alignment and clinical correlation analysis
