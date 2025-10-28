# Causal Metric Tensor: Summary of Findings

**Last updated:** October 27, 2025

This document summarizes our discoveries about the causal metric tensor M = Cov(γ)⁻¹ and the geometric structure of token space.

---

## What Is The Causal Metric?

**Reference:** Park et al. (2024) - "The Linear Representation Hypothesis and the Geometry of Large Language Models"

The causal metric tensor is defined as:
```
M = Cov(γ)⁻¹
```

Where:
- γ is the unembedding matrix (vocab_size × hidden_dim)
- Cov(γ) is the covariance matrix of token vectors
- M defines distances via: d(v, w) = √((v - w)ᵀ M (v - w))

**Units:** We measure distances in **logometers** (logo = word/language + meters).

**Key insight:** The causal metric captures the model's learned probability structure, revealing the "true" geometry of semantic space — not the Euclidean geometry of the parameter space.

---

## The Astronomy Framing

**Critical paradigm shift:** Token space is a **discrete point cloud** (152,936 stars), not a smooth manifold.

We study it using **observational methods** (statistics, clustering, sampling) rather than differential geometry (derivatives, geodesics, smooth surfaces).

Think **astronomy**, not calculus.

### The Token Galaxy

- **152,936 tokens** scattered in 2560-dimensional space
- Form a thin **shell** structure around **~54 logometers** from origin (mean causal norm)
- Radial variation: **4% CV** under causal metric

**Important:** The shell structure itself is expected in high dimensions (curse of dimensionality—random points concentrate at the surface). But our tokens show **100× MORE radial spread** than random uniform distribution (which has CV ≈ 0.04%). This spread is emergent structure, not randomness.

**The magnification story:**
- Start with Euclidean norms: mean ~1.13, CV 15.5% (moderate spread)
- Apply causal metric: mean ~54, CV 4% (tighter relative spread)
- **But**: The 50× scaling AMPLIFIES absolute differences
- Tokens that scale less (40×) vs. more (60×) end up further apart in absolute terms
- Result: radially structured, not uniformly scaled

---

## Basic Geometric Properties (04.1)

### Effective Dimensionality
- **~52%** effective dimensionality via participation ratio
- PR = (Σλᵢ)² / Σ(λᵢ²) ≈ 1333 active dimensions (out of 2560)
- Space is **anisotropic** — not all directions are equal

### Non-Euclidean Distance
- **375,000% deviation** from Euclidean geometry
- ||M - I||_F is enormous
- Space is **strongly non-Euclidean**

### Token Cloud Extent
- Diameter: **~112 logometers**
- Typical pairwise distances measured via sampling
- Finite extent (not infinite scatter)

---

## Discrete Curvature (04.2)

**Method:** Forman-Ricci curvature on k-NN graph using causal distances

**Key finding:**
- **Positive curvature** κ ≈ 26.7
- Indicates **sphere-like local clustering**
- Space is neither flat nor hyperbolic
- Tokens cluster together rather than spread uniformly

**Interpretation:** The token cloud has positive curvature under the causal metric, suggesting local "clumping" of semantically related tokens.

---

## Community Structure (04.3)

**Method:** Louvain community detection on k-NN graph (n=8,000 sample)

**Key findings:**
- **4 roughly equal communities** (28%, 27%, 25%, 20% of sample)
- Modularity 0.47 indicates genuine clustering (not random)
- Graph is fully connected (single continuous manifold)
- Surprising symmetry in size distribution

**Interpretation:** Semantic space has distinct clusters, but they're balanced (not power-law or hierarchical). The model didn't create a simple taxonomy — it found a more symmetric organization.

**Uncertainty:** The exact number "4" has error bars. Sampling effects (8k/152k ≈ 5%) and Louvain's resolution parameter mean the true answer is likely **between 2-6 major communities** at this hierarchical level. Could also have:
- **Sub-communities** within each major cluster (hierarchical structure we didn't detect)
- **Tiny specialized communities** (emoji, math symbols, rare languages) undersampled

**High confidence**: Small number of large, balanced communities. NOT power-law, NOT thousands of micro-clusters, NOT one giant blob.

**Low confidence**: Exact count, presence of hierarchical sub-structure.

---

## Magnitude Transformation (04.5b)

**Question:** How does the causal metric transform token magnitudes compared to Euclidean?

### Euclidean Norms (Baseline)
- Mean: ~1.13 units
- CV: **15.5%** (moderate variation)
- Tokens are **not pre-normalized** (emergent concentration)

### Causal Norms
- Mean: **~54 logometers**
- CV: **4%** (very tight!)
- ~50× scaling from Euclidean
- Range: 21–85 logometers

**Key insight:** The causal metric applies **nearly uniform radial scaling** — magnitudes become much more consistent.

---

## Angular Warping (04.5c)

**Question:** How does the causal metric transform angles between token pairs?

**Method:** Sample 10,000 random token pairs, measure angles in both metrics

### Key Findings
- **Mean angular distortion:** Δθ = -4.32° (systematic compression)
- **Std:** 2.20° (consistent, not chaotic)
- **CV:** 51% (moderate spread relative to mean)
- **Range:** -28.7° to +1.9° (asymmetric, mostly negative)

**Interpretation:**
- Causal metric applies **systematic angular compression**
- NOT conformal (angles not preserved)
- NOT chaotic (predictable distortion)
- Space is "stretched radially, pinched angularly"

### Arc Displacement
At radius ~50 logometers, a 4.3° angular shift corresponds to:
- **Arc length:** s = r × Δθ ≈ 50 × 0.0754 ≈ **3.8 logometers**
- Comparable to original Euclidean magnitude (~1 unit)!
- Small angles → large tangential displacements at extreme radii

---

## Quasar Tokens (04.6a)

**Question:** What are the most distant tokens (outliers in causal space)?

### Top 3 Quasars
1. **`<|endoftext|>`** — 85.3 logometers (1.58× mean)
2. **`\n` (newline)** — 82.1 logometers
3. **`\u200b\u200b` (zero-width space)** — 81.2 logometers

### Pattern
**Structural/meta-linguistic tokens** dominate the extreme distances:
- Whitespace: `\n`, `\u200b\u200b`, `\xa0`
- Control: `<|endoftext|>`
- Fragments: camelCase suffixes, rare programming tokens

**Interpretation:** The model pushes structural markers (boundaries, formatting) to extreme distances, separating them from semantic content.

### Tokens Near Origin
- **Rare/unused tokens** at ~21 logometers (0.39× mean)
- Korean/Chinese characters, emoji, replacement characters
- Undertrained → weak vectors

---

## Quasar Constellation (04.6a2)

**Question:** Are the top quasars angularly clustered or scattered?

### Pairwise Angles
- **endoftext ↔ newline:** 69.72° Euclidean / 79.62° causal (+9.90°)
- **endoftext ↔ zero-width:** 93.15° Euclidean / 89.28° causal (-3.87°)
- **newline ↔ zero-width:** 96.31° Euclidean / 88.41° causal (-7.90°)

**Key findings:**
- **Widely separated** — nearly orthogonal (~70-90°)
- **NOT a constellation** — they're independent reference directions!
- Average distortion: **-0.62°** (vs typical -4.32°)

**Interpretation:**
- Structural tokens are **geometrically special** — they barely warp under causal metric
- We have **three nearly orthogonal axes** in token space
- Different structural roles → different directions:
  - `<|endoftext|>`: Termination (document boundaries)
  - `\n`: Segmentation (line/paragraph breaks)
  - `\u200b\u200b`: Formatting (invisible layout control)

---

## Main Sequence Orthogonality (04.6a3)

**Question:** Are typical semantic tokens (main sequence at ~54 logometers) orthogonal to structural tokens (quasars)?

### Main Sequence Tokens
- Top 100 tokens closest to mean causal norm
- Examples: "UniformLocation", "Downloader", " Pluto", " Intelligence", " prejudice"
- Mix of technical, multilingual, semantic content
- NOT ultra-common function words (those aren't at the exact mean)

### Projection Measurements

**Euclidean space (training coordinates):**
- Quasars: 0.1173 mean projection
- Control (random typical tokens): 0.0916
- **Ratio: 1.28** → Quasars are **LESS orthogonal** than random
- Main sequence tokens are **aligned** with structural markers

**Causal space (learned probability metric):**
- Quasars: 2.8974 mean projection
- Control: 8.2217
- **Ratio: 0.35** → Quasars are **3× MORE orthogonal** than random!
- Structural tokens occupy a **geometrically distinct subspace**

### Initial Interpretation (Revised by 04.6a4 & 04.6a5)

**What we thought:** The model disentangles structural and semantic information — an emergent functional principle.

**What's actually happening:** The 3 extreme quasars are orthogonal to the main sequence primarily because they're **geometrically isolated by distance**, not because "structure ⊥ content" is a general organizing principle.

---

## Structural Subspace Test (04.6a4)

**Question:** Does the orthogonality finding generalize to ALL structural tokens?

**Method:** Identified ~3,000 structural tokens (special tokens, whitespace, control characters, formatting) and measured their causal orthogonality to main sequence.

**Result:** **Ratio = 1.0165** (almost perfect match to control)

**Interpretation:** Most structural tokens are **NOT** causally orthogonal to semantic content. They live mixed in with semantic tokens. Only the extreme outliers (the 3 brightest quasars) show strong orthogonality.

**Conclusion:** "Structure ⊥ content" is **not** a general principle. It's specific to the most distant boundary markers.

---

## Angular Census (04.6a5)

**THE REAL DISCOVERY**

**Question:** Is `<|endoftext|>` encoding a special semantic direction, or is it just geometrically isolated?

**Method:** Compute causal cosine similarity between `<|endoftext|>` and all 151,936 tokens.

### Results

**Distribution statistics:**
- Mean cosine: **-0.000002** (essentially zero)
- Std: **0.018205** (vs. random expectation 0.019764)
- **99.9% orthogonal** (|cos| ≤ 0.1)
- Only 80 tokens (0.05%) show mild parallel alignment
- Only 13 tokens (0.009%) show mild antiparallel alignment

**Comparison to random:** The distribution is **indistinguishable from random high-dimensional vectors**.

### The Profound Truth

**`<|endoftext|>` is orthogonal to EVERYTHING.**

It's not encoding a special direction. It's not the "opposite" of semantic content. It's **geometrically random** — just pointing in an arbitrary direction that happens to be **very far away** (85 logometers vs. 54 mean).

**Distance-induced orthogonality:** In high-dimensional space, distant points are nearly orthogonal to everything near the origin. The quasars are orthogonal to the main sequence because they're **isolated in deep space**, not because they encode functional structure.

### What Tokens Show ANY Alignment?

The few tokens with cos > 0.1:
1. `<|endoftext|>` itself (cos=1.0)
2. `<|fim_middle|>` (cos=0.33) — another special token
3. 32 consecutive newlines (cos=0.19)
4. 16 consecutive newlines (cos=0.19)
5. Rare multilingual characters

**These are all quasar candidates** — other distant outliers in the same sparse region of token space.

### Revised Understanding

**The quasars aren't special because they're structural.**

They're special because they're **alone out there** — distant lighthouses visible from anywhere (high norm = "bright"), but not aligned with any semantic axis. They occupy their own sparse region far from the semantic mainland, where everything is mutually orthogonal by sheer geometry

---

## Implications (Revised)

### For Understanding Token Space Geometry

**The "true" geometry isn't the parameter space (Euclidean) — it's the probability space (causal metric).**

Token space has a **core-periphery structure**:

- **Core (main sequence)**: ~54 logometers, dense cluster of semantic tokens (99.9% of vocabulary)
- **Periphery (quasars)**: 80+ logometers, sparse outliers that are geometrically isolated
- **Distance-induced orthogonality**: Periphery tokens are orthogonal to everything (including each other) by high-dimensional geometry

**Not a functional separation** (structure vs. content), but a **distance effect** (core vs. periphery).

### For Steering

**Off-manifold = heading toward the periphery**

When perplexity spikes during steering, you're likely drifting:
- **Away from the dense core** (main sequence at ~54 logometers)
- **Toward sparse periphery** (quasars at 80+ logometers)
- Into regions where tokens are isolated and geometrically random

**Steering safety:** Stay within typical causal norm range (45-65 logometers). Going beyond ~70 logometers means entering the sparse periphery where few tokens exist.

### For Manifold Boundaries

**Revised definition:**
- **Manifold** ≈ dense core (causal norm 45-65 logometers)
- **Main sequence** ≈ modal radius (~54 logometers)
- **Off-manifold** ≈ sparse periphery (>70 logometers) OR near-origin rare tokens (<30 logometers)

The manifold isn't defined by semantic vs. structural content — it's defined by **token density in causal space**.

### Why Quasars Cause Perplexity Spikes

When steering pushes activations toward `<|endoftext|>`:
- Not because you're "encoding structure instead of content"
- But because you're in a **sparse region** where almost no tokens exist
- Model has no nearby tokens to predict → perplexity spike

The geometry is simpler than we thought: **stay where the tokens are dense**.

---

## Open Questions

1. **Layer-wise variation:** How does the causal metric change across layers?
2. **Causal steering implementation:** How to practically apply steering using M?
3. **Other quasar types:** Are there non-structural outliers we've missed?
4. **Manifold-aware α selection:** Can we predict optimal steering coefficient from causal geometry?
5. **Community interpretation:** What do the 4 semantic clusters represent?
6. **Full-universe Forman-Ricci curvature (low priority):** Current estimate κ ≈ 24-27 from n=8000 sample, likely converges ~26-30. Computing exact value on all 152k tokens is feasible (~$5 on H200, or overnight local compute with chunked distances), but unclear if precision gain justifies cost. The qualitative finding (positive curvature → sphere-like clustering) is established via multiple methods (Hopkins statistic, modularity, community detection). More valuable would be: regional/directional curvature variations, layer-wise curvature evolution, or comparison across different distance metrics. **Noted here specifically so we don't forget we considered and deprioritized this.**

---

## Notebooks Reference

- **03:** Extract causal metric tensor M
- **04.1:** Basic properties (dimensionality, extent, non-Euclidean distance)
- **04.2:** Forman-Ricci curvature (positive, sphere-like)
- **04.3:** Community detection (4 equal clusters)
- **04.4:** UMAP visualization (2D/3D projections)
- **04.5:** Initial norm comparison (Euclidean vs causal)
- **04.5b:** Euclidean baseline (unnormalized, 15.5% CV)
- **04.5c:** Angular warping (-4.32° systematic compression)
- **04.6:** Quasar framework (design doc)
- **04.6a:** Deep space quasars (find most distant tokens)
- **04.6a2:** Quasar angles (nearly orthogonal constellation)
- **04.6a3:** Main sequence orthogonality (initial finding)
- **04.6a4:** Structural subspace test (general tokens NOT orthogonal)
- **04.6a5:** Angular census (**THE REAL DISCOVERY** - distance-induced orthogonality)

---

## Final Thoughts

The causal metric reveals that language models organize semantic space with a **core-periphery structure**:

**Core (99.9% of tokens):** Dense semantic cluster at ~54 logometers, where tokens interact and predict each other.

**Periphery (0.1% of tokens):** Sparse outliers at 80+ logometers, geometrically isolated from everything by distance alone.

The profound finding from 04.6a5: **Extreme distance creates orthogonality**. Quasars aren't special because they encode "structure" — they're special because they're **alone in deep space**, far from the semantic mainland where language actually happens.

The geometry we see in M is the geometry of **language as probability**, not language as parameters. And that geometry is simpler than we first thought: a dense core where meaning lives, surrounded by a sparse periphery of isolated outliers.
