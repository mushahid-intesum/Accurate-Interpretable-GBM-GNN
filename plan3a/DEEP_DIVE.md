# Plan 3a — Technical Deep Dive

A complete mathematical and implementation walkthrough of every component.

---

## Table of Contents

1. [Concept Bottleneck Model (CBM)](#1-concept-bottleneck-model-cbm)
2. [Sheaf Hypergraph Neural Network](#2-sheaf-hypergraph-neural-network)
3. [HECRL — Inter-Concept Refinement](#3-hecrl--inter-concept-refinement)
4. [Internal Model Flow (End-to-End)](#4-internal-model-flow-end-to-end)
5. [All Evaluation Metrics](#5-all-evaluation-metrics)

---

## 1. Concept Bottleneck Model (CBM)

### The Problem CBMs Solve

A standard neural network maps input → prediction with no human-readable intermediate step:

```
MRI Patches → [black box] → "Patient survives 400 days"
```

A clinician cannot ask *why*. Was it the tumor's enhancement pattern? Its location? The diffusion characteristics?

A **Concept Bottleneck Model** forces a detour through interpretable concepts:

```
MRI Patches → [encoder] → Concepts → [classifier] → "Patient survives 400 days"
                            ↑
                     Enhancement = 3.2
                     FLAIR z-score = 1.8
                     Heterogeneity = 0.7
                     ...
```

The classifier **can only see concept values**, never the raw embeddings. This is a hard information barrier — the model has no choice but to express its reasoning through concepts.

### How Our CBM Works (concept_bottleneck.py)

**Step 1: Concept Prediction**

Each node (MRI patch) has a 128-dim embedding from the SheafHGNN. The `ConceptPredictor` maps this to 8 scalar concept values:

```
node_embedding (128-dim)
    │
    ▼
SharedTrunk: Linear(128→64) + LayerNorm + GELU
    │
    ├──→ Head₁: Linear(64→32) + GELU + Linear(32→1) → c₁ (enhancement)
    ├──→ Head₂: Linear(64→32) + GELU + Linear(32→1) → c₂ (FLAIR)
    ├──→ Head₃: ...                                   → c₃ (T2)
    │    ...
    └──→ Head₈: ...                                   → c₈ (spatial)
```

Each concept gets its **own prediction head** — this encourages disentangled representations where each head specializes in one clinical property.

**Step 2: Self-Supervised Concept Loss**

The predicted concepts are compared to precomputed ground truth (from `patch_extraction.py`):

```
L_concept = (1/N) Σᵢ Σⱼ mⱼ · (ĉᵢⱼ - cᵢⱼ)²
```

Where:
- `ĉᵢⱼ` = predicted concept j for node i
- `cᵢⱼ` = precomputed concept j for node i  
- `mⱼ` = mask (0 for c₇ which has no ground truth, 0 for concepts from missing modalities)

This is **self-supervised** because the targets come from the raw MRI intensities, not from manual annotation. For example, c₁ (enhancement ratio) = `log(1 + T1_post / T1_pre)` — computed directly from the DICOM pixel values.

**Step 3: The Bottleneck**

After concept prediction (and optional HECRL refinement), the bottleneck output is:

```python
# residual_bypass = False (default — strict bottleneck)
bottleneck_output = concepts_refined        # shape: (N, 8)

# residual_bypass = True (E1/E2 baselines — breaks interpretability)
bottleneck_output = cat([concepts, embeddings])  # shape: (N, 8+128)
```

When `residual_bypass=False`, the downstream survival head literally receives only 8 numbers per node. Every prediction must be traceable to these 8 concepts.

**Why This Matters for Research**

Traditional post-hoc methods (GNNExplainer, SHAP) explain *after* the model decides. The ICLR 2026 paper showed these can be **degenerate** — producing explanations that look valid but don't reflect the model's actual reasoning. The CBM avoids this by construction: there is no hidden reasoning path that bypasses concepts.

---

## 2. Sheaf Hypergraph Neural Network

### Why Hypergraphs (Not Regular Graphs)

A regular graph edge connects exactly 2 nodes. A **hyperedge** connects any number of nodes simultaneously. This matters for brain tumor analysis because:

- A tumor region is not pairwise — it's a *group* of patches sharing a biological property
- In a regular GNN, information passes only between directly connected pairs
- In a hypergraph, a single hyperedge can broadcast information across an entire tissue region at once

### What is a "Sheaf"?

In a standard Hypergraph Neural Network, all information flows identically through every connection. A **sheaf** assigns a unique linear transformation to each (node, hyperedge) pair:

```
Standard HGNN:   node_i → aggregate → hyperedge_e → aggregate → node_j
                 (same everywhere)

Sheaf HGNN:      node_i →[F_{i⊥e}]→ hyperedge_e →[F_{e⊥j}]→ node_j
                 (unique learned transforms per connection)
```

`F_{v⊥e}` is a learned matrix that controls *how* node v's features are projected when contributing to hyperedge e. Different sheaf maps allow:
- A "tumor core" hyperedge to focus on enhancement features
- An "edema" hyperedge to focus on FLAIR features
- A "DTI tract" hyperedge to focus on diffusion features

### Internal Operation (sheaf_hgnn.py, lines 98–160)

One layer of SheafHGNN performs four steps:

**Step 1: Vertex → Hyperedge (with sheaf map)**

```python
x_transformed = F_v2e(x)  # Linear(128→128, no bias) — the sheaf map

# For each hyperedge e, average the transformed features of its member nodes
edge_features[e] = (1/|e|) Σ_{v∈e} x_transformed[v]
```

This is a scatter-mean: each hyperedge collects the sheaf-projected features of all nodes it contains.

**Step 2: Hyperedge → Vertex (with sheaf map)**

```python
edge_transformed = F_e2v(edge_features)  # Another sheaf map

# For each node v, sum the transformed features from all its hyperedges
node_updates[v] = Σ_{e∋v} edge_transformed[e]
```

Each node receives aggregated information from all hyperedges it belongs to.

**Step 3: Sheaf Laplacian Normalization**

```python
# Normalize by node degree (number of hyperedge memberships)
inv_sqrt_degree = 1 / √(degree[v])
node_updates[v] = node_updates[v] × inv_sqrt_degree
```

This is the simplified sheaf Laplacian: `Δ_F = I - D_v^{-1/2} · L_F · D_v^{-1/2}`. It prevents high-degree nodes from dominating.

**Step 4: Transform + Normalize + Activate**

```python
x_out = GELU(LayerNorm(Θ · node_updates))  # Θ = learned weight matrix
```

### Multi-Layer Stacking (3 layers)

The full SheafHGNN stacks 3 layers with residual connections:

```
x₀ = PatchEncoder(raw_patches)                  # (N, 128)
x₁ = x₀ + SheafHGNNLayer₁(x₀)                  # 1-hop neighborhood
x₂ = x₁ + SheafHGNNLayer₂(x₁)                  # 2-hop neighborhood
x₃ = x₂ + SheafHGNNLayer₃(x₂)                  # 3-hop neighborhood

# Multi-scale fusion: concatenate ALL layer outputs
x_multi = cat([x₀, x₁, x₂, x₃])                # (N, 512)
x_final = Linear(512→128) + LayerNorm + GELU     # (N, 128)
```

The multi-scale fusion is important: it lets the model use features from different receptive fields simultaneously. Layer 1 sees immediate neighbors, layer 3 sees the whole tumor region.

### Our Hyperedge Construction (hypergraph.py)

We build two types of hyperedges:

1. **Topological (δ-ball)**: patches within spatial radius δ=2.5 share a hyperedge → captures local anatomy
2. **Feature-based (top-k)**: patches with most similar features (k=9 nearest) share a hyperedge → captures functional similarity across spatial distance

---

## 3. HECRL — Inter-Concept Refinement

### The Problem HECRL Solves

The ConceptPredictor predicts each concept independently. But concepts are **not independent** in reality:

- High enhancement (c₁) + high perfusion → likely active tumor
- High FLAIR (c₂) + low enhancement → likely edema, not tumor core
- High DTI FA (c₅) in a high-enhancement region → unexpected, possibly artifact

HECRL (Hypergraph-Enhanced Concept Representation Learning, from HyperCBM) enforces these inter-concept dependencies.

### How HECRL Works (concept_bottleneck.py, lines 85–149)

HECRL treats the 8 concepts as a **sequence of tokens** and applies self-attention:

**Step 1: Embed each concept as a token**

```python
# Each concept value is a scalar → project to 32-dim
# concepts: (N, 8) → concepts.unsqueeze(-1): (N, 8, 1)
c_tokens = Linear(1→32)(concepts)  # (N, 8, 32)
```

Now each concept is a 32-dimensional token. The 8 tokens form a "concept sequence" per node.

**Step 2: Multi-Head Self-Attention across concepts**

```python
# Q = K = V = c_tokens
# 2-head attention, each head has dim 16
attn_out = MultiHeadAttention(c_tokens, c_tokens, c_tokens)  # (N, 8, 32)
c_refined = LayerNorm(c_tokens + attn_out)                    # (N, 8, 32)
```

The attention mechanism lets each concept attend to every other concept. The attention weights learn patterns like:
- "When computing the refined c₁ (enhancement), pay attention to c₂ (FLAIR) and c₆ (heterogeneity)"
- "c₈ (spatial location) should modulate all other concepts"

**Step 3: Project back to scalar values**

```python
refined = Linear(32→1)(c_refined).squeeze(-1)  # (N, 8)
```

### Why "Hypergraph-Enhanced"?

The original HyperCBM paper builds an actual concept-level hypergraph where concepts are nodes and hyperedges represent co-occurrence patterns. Our implementation approximates this with multi-head self-attention (which is mathematically equivalent to a complete hypergraph with soft edges) because we only have 8 concepts — full attention is computationally cheaper than sparse hypergraph operations at this scale.

---

## 4. Internal Model Flow (End-to-End)

Here is the complete data flow for one patient, with tensor shapes:

```
INPUT: UPENN-GBM-00001
  - 4,105 MRI patches, each 16×16 across 6 modalities
  - 18-dim clinical vector (age, gender, IDH1, MGMT, KPS, GTR)
  - Survival: 960 days, deceased

STAGE 1: SheafHGNN
  node_features: (4105, 1536) ← 6 × 16 × 16 flattened
     ↓ PatchEncoder MLP
  x₀: (4105, 128)
     ↓ 3× SheafHGNNLayer (7,829 hyperedges)
  x_final: (4105, 128)                    [~0.15s]

STAGE 2: ConceptBottleneck
  x_final: (4105, 128)
     ↓ ConceptPredictor (8 disentangled heads)
  concepts_raw: (4105, 8)
     ↓ Replace c₇ with boundary_head(x_final)
  concepts_raw: (4105, 8) [c₇ now graph-learned]
     ↓ HECRL self-attention
  concepts_refined: (4105, 8)              [bottleneck barrier]
     ↓ concept_loss = MSE(refined, precomputed_targets)

STAGE 2→3: Concept Pooling
  concepts_refined: (4105, 8)
     ↓ Linear(8→128) + LayerNorm + GELU
  concept_graph: (4105, 128)
     ↓ Attention-weighted pooling
  graph_embed: (1, 128)                    [entire tumor = 1 vector]

STAGE 3: MultiModalFusion
  graph_embed: (1, 128)    clinical_raw: (1, 18)
     │                        ↓ ClinicalEncoder MLP
     │                     clin_embed: (1, 128)
     │                        │
     ↓──── DynamicWeighting ──↓
     w_img=0.62, w_clin=0.38          [per-patient balance]
     │                        │
     ↓ × w_img               ↓ × w_clin
     │                        │
     └──── CrossAttention ────┘
                 ↓
  fused: (1, 128)

STAGE 4: SurvivalHead
  fused: (1, 128)
     ↓ Linear(128→64) + GELU + Dropout
     ↓ Linear(64→32) + GELU + Dropout
     ↓ Linear(32→4)
  hazard_logits: (1, 4)                   [one per time bin]
     ↓ sigmoid
  hazard_probs: [0.12, 0.28, 0.45, 0.67] [P(event in bin k)]
     ↓ S(t) = Π(1-hₖ), risk = 1-S(t)
  risk_score: 0.87                         [high risk → short survival]

LOSS: Kendall multi-task weighting
  L = (1/2σ₁²)·L_survival + (1/2σ₂²)·L_concept + log(σ₁·σ₂)
  where σ₁, σ₂ are learned (start equal, model adjusts during training)
```

### Dynamic Weighting (Fusion, in detail)

The fusion module prevents **modality collapse** — where the model ignores one input:

**Mono-confidence**: How reliable is each modality alone?
```
w_img_mono  = σ(Linear(img_embed))     ∈ (0, 1)
w_clin_mono = σ(Linear(clin_embed))    ∈ (0, 1)
```

**Holo-confidence**: How do the modalities interact?
```
w_img_holo  = log(w_img_mono) / log(w_img_mono × w_clin_mono)
w_clin_holo = log(w_clin_mono) / log(w_img_mono × w_clin_mono)
```

This is from MRePath Eq. 7–9. The intuition: if one modality is very confident (mono≈1), its holo-weight increases because it's contributing more to the joint confidence.

**Final weights**: `softmax(mono + holo)` → guaranteed to sum to 1.

---

## 5. All Evaluation Metrics

### 5.1 Primary Task Metric: Harrell's C-Index

**What it measures**: Does the model correctly rank patients by survival risk?

**Formal definition**:
```
C-Index = (concordant pairs + 0.5 × tied pairs) / total comparable pairs
```

**How it works (task_metrics.py, lines 12–62)**:

For every pair of patients (i, j) where patient i died before patient j:
- If model predicts `risk(i) > risk(j)` → **concordant** ✓
- If model predicts `risk(i) < risk(j)` → **discordant** ✗
- If model predicts `risk(i) = risk(j)` → **tied** (counts as 0.5)

Only "comparable" pairs are counted — pairs where we *know* who died first. A censored patient (still alive at last follow-up) is only compared against patients who died *before* the censoring time.

**Rationale**: C-Index is the standard survival analysis metric because:
1. It handles **right-censored data** — patients who are still alive at study end
2. It only asks "who dies first?", not "when exactly?" — more clinically useful
3. Range [0, 1]: 0.5 = random coin flip, 1.0 = perfect ranking
4. It's **threshold-free** — doesn't require choosing a cutoff

**Target**: C-Index > 0.65 is considered clinically useful for GBM.

### 5.2 Survival Loss: Discrete-Time NLL

**What it measures**: How well the model predicts *when* events occur.

**How it works (full_model.py, lines 58–116)**:

Time is discretized into 4 bins (boundaries set by quartiles of event times in training data). The model outputs a hazard probability `hₖ` for each bin — "probability that the event occurs in bin k, given survival past bin k-1."

For a patient who died in bin k:
```
L = -log(hₖ) - Σ_{j<k} log(1 - hⱼ)
```

For a censored patient (still alive) at bin k:
```
L = -Σ_{j≤k} log(1 - hⱼ)
```

**Rationale**: 
- The first term says "the event *did* happen in bin k — penalize low hₖ"
- The second term says "the patient *survived* all earlier bins — penalize high hⱼ for j < k"
- Right-censored patients contribute only the survival terms — we know they survived *at least* this long but don't know when they die

**Why discrete-time (not Cox PH)?**: Cox regression assumes proportional hazards — that risk ratios between patients are constant over time. Brain tumors violate this (a patient might respond to treatment temporarily). Discrete-time NLL makes no such assumption.

### 5.3 Concept Metrics: Per-Concept MSE and Pearson r

**What they measure**: How accurately the bottleneck reconstructs each concept.

**MSE** (Mean Squared Error):
```
MSE_j = (1/N) Σᵢ (ĉᵢⱼ - cᵢⱼ)²
```
Lower = better. Measures absolute reconstruction accuracy.

**Pearson r** (correlation):
```
r_j = Cov(ĉⱼ, cⱼ) / (σ(ĉⱼ) · σ(cⱼ))
```
Range [-1, 1]. Measures whether the predicted concept *tracks* the true concept, even if scaled differently.

**Rationale**: We need both metrics because:
- MSE penalizes scale errors (predicted 5.0, actual 2.0 is bad)
- Pearson r ignores scale, catches ranking errors (predicted [1,2,3], actual [3,2,1] gives r = -1.0)
- A concept with low MSE but low r is memorizing the mean
- A concept with high MSE but high r is correct in ranking but needs calibration

**Why exclude c₇?**: Concept 7 (boundary complexity) has no precomputed ground truth — it's learned directly from the graph structure via a dedicated head. Including it in metrics would show artificially perfect correlation.

### 5.4 Faithfulness Metrics (ICLR 2026 Audit)

These four metrics answer: **are the model's explanations genuinely faithful, or degenerate?**

An "explanation" is the top 20% most activated nodes (patches with highest concept activation L2 norm). Call this set **R** (explanation) and its complement **G∖R** (the rest).

#### EST — Extension Sufficiency Test

**What it detects**: Anchor-set degeneracy — explanations that work only because they contain a structural shortcut.

**Procedure**:
```
1. Start with explanation R (top 20% nodes)
2. For s = 1 to 50:
     a. Sample random subset S ⊂ (G∖R)
     b. Build supergraph R' = R ∪ S
     c. Predict on R'
     d. Record |pred(R') - pred(G)|
3. EST = max shift across all 50 samples
4. PASS if EST < threshold (0.1)
```

**Rationale**: If the explanation R truly captures the model's reasoning, adding irrelevant nodes shouldn't change the prediction. But if R contains an "anchor set" — a minimal structural pattern that forces a specific prediction regardless of what else is present — then adding nodes *will* shift the prediction because the anchor interacts differently with different neighborhoods.

A low EST score means: "No matter what I add to this explanation, the prediction stays the same" → the explanation is genuinely sufficient.

#### Fid⁻ — Fidelity Minus

**What it measures**: Does the explanation alone reproduce the full-graph prediction?

**Procedure**:
```
1. Remove all complement nodes (keep only R)
2. Filter hyperedges to only those within R
3. Predict on R alone
4. Fid⁻ = |pred(R) - pred(G)|
5. PASS if Fid⁻ < threshold (0.1)
```

**Rationale**: If the explanation contains the model's true reasoning, running the model on *just* the explanation should give the same answer as running it on the full graph. A high Fid⁻ means the model needs information outside the explanation → the explanation is incomplete.

#### RFid⁻ — Randomized Fidelity Minus

**What it measures**: How sensitive is the prediction to random perturbation of the complement?

**Procedure**:
```
1. For each connection involving a complement node:
     Drop it with probability p = 0.9
2. Keep all explanation connections intact
3. Predict on the perturbed graph
4. Record |pred(perturbed) - pred(G)|
5. Average over 20 runs
6. PASS if mean shift < threshold (0.1)
```

**Rationale**: Fid⁻ is an extreme test (remove ALL complement). RFid⁻ is softer — it randomly damages the complement. If the explanation is faithful, this damage shouldn't matter because the important information is in R. RFid⁻ catches cases where Fid⁻ passes due to the model's robustness rather than the explanation's quality.

#### Suf — Sufficiency

**What it measures**: If we keep the explanation but replace everything else with noise, does the prediction hold?

**Procedure**:
```
1. Keep explanation node features intact
2. Replace complement node features with Gaussian noise
     (matched to feature mean and std)
3. Keep the full hypergraph structure
4. Predict
5. Suf = |pred(noised) - pred(G)|
6. PASS if Suf < threshold (0.1)
```

**Rationale**: This tests a different failure mode than Fid⁻. In Fid⁻, we remove complement *structure* (edges). In Suf, we keep the structure but destroy complement *content* (features). If the explanation is sufficient, the complement's feature values should be irrelevant. A high Suf score means the model is secretly using complement features → the explanation misses something.

#### Rejection Ratios

**What they measure**: Across all patients, what fraction of explanations fail each test?

```
Rejection_EST = (# patients where EST fails) / (# patients audited)
```

**Rationale**: A single patient might have an unusual graph structure that causes one metric to fail. Rejection ratios give the population-level picture:
- 0% rejection = all explanations pass → strong faithfulness
- >50% rejection = most explanations are degenerate → the model has shortcut problems
- The **overall rejection** requires ALL four tests to pass — a single failure rejects

### 5.5 Multi-Task Loss Weighting (Kendall)

**What it does**: Automatically balances survival loss vs concept loss during training.

```
L = (1/2σ₁²)·L_survival + (1/2σ₂²)·L_concept + log(σ₁·σ₂)
```

Where σ₁, σ₂ are learned parameters (initialized to 1.0).

**Rationale**: The two tasks have different scales (survival NLL ≈ 1.2, concept MSE ≈ 0.5 initially). Fixed weighting (e.g., 0.5 + 0.5) would let the larger-scale task dominate. Kendall's uncertainty weighting (CVPR 2018) interprets σ as task uncertainty:
- High σ → low weight (uncertain task gets less gradient pressure)
- The `log(σ₁·σ₂)` regularizer prevents σ from going to infinity (which would zero out both losses)
- During training, the model learns that survival prediction is inherently noisier → increases σ₁ → allocates proportionally more gradient to concept reconstruction

---

## Summary: Why Each Piece Exists

| Component | Without It | With It |
|-----------|-----------|---------|
| **CBM** | Black-box prediction, can't explain why | Every prediction traceable to 8 concepts |
| **Sheaf maps** | All hyperedges aggregate identically | Different hyperedges focus on different features |
| **HECRL** | Concepts predicted independently | Concepts refined with inter-concept consistency |
| **Dynamic weighting** | One modality dominates (usually imaging) | Per-patient adaptive balance |
| **NLL survival loss** | Can't handle censored patients | Properly models right-censoring |
| **C-Index** | Would need arbitrary survival cutoff | Threshold-free ranking evaluation |
| **EST audit** | Could produce degenerate explanations | Formally verifies faithfulness |
| **Kendall weighting** | Must manually tune loss balance | Auto-balances survival vs concept tasks |
