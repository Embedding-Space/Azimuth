# FLRW-Style Metric Expansion in Transformers

**Date:** October 29, 2025
**Status:** Hypothesis / Phenomenological Model
**Confidence:** Intriguing but untested

---

## Executive Summary

We observe that token activation norms grow exponentially through transformer layers (~77× from L=0 to L=35 in Qwen 3-4B). When normalized by a layer-dependent scale factor a(L), representations become approximately stationary in "comoving coordinates" (89.8% reduction in coefficient of variation).

**Hypothesis:** The causal metric tensor is layer-dependent, transforming from Euclidean at embeddings to the Park et al. causal metric at unembedding via FLRW-style expansion:

```
M(0) = δᵢⱼ                           (Euclidean, by definition)
M(max(L)) = Cov(γ)⁻¹                 (Causal, from Park et al.)
M(L) = [a(L) / a(max(L))]² M(max(L)) (Scaled interpolation)
```

where a(L) is the empirically measured scale factor at layer L.

**Implications if true:**
- Transformers expand semantic space rather than moving representations through it
- LayerNorm divides by a(L), keeping representations in comoving coordinates
- Inter-layer "redshift" z(L₁→L₂) = a(L₂)/a(L₁) - 1 measures semantic distance
- Steering vectors scale as V(L) = a(L) × V_comoving

---

## Observational Evidence

### 1. Exponential Norm Growth (Notebook 08.2)

**Data:** 512 tokens from one Wikipedia text, causal norms measured at all 36 layers.

**Finding:** Token norms grow approximately exponentially:
- Layer 0: ~400-600 logometers
- Layer 35: ~30,000-50,000 logometers
- Mean growth: 76.9× over 36 layers

When plotted on log scale, trajectories are approximately linear → exponential growth.

### 2. Scale Factor a(L) (Notebook 08.3)

**Extraction:** a(L) = <||activation(L)||_M> / <||activation(0)||_M>

**Fit:** a(L) ≈ exp(H₀ × L) where H₀ = 0.1288 ± 0.0007 [1/layer]
- R² = 0.9728 (good fit despite complex H(L) structure)
- Total expansion: a(35) = 76.9×

**Hubble parameter H(L) = (1/a) da/dL shows multi-epoch structure:**
1. Early inflation (L=0): H ≈ 0.45
2. Deceleration (L=1-7): H drops to 0.02
3. Steady expansion (L=8-17): H ≈ 0.02-0.05
4. Re-acceleration (L=18-23): H climbs to 0.27
5. Late plateau (L=24-33): H ≈ 0.13-0.19
6. Final contraction (L=34-36): H → -0.21 (Big Crunch before unembedding)

### 3. Comoving Coordinates (Notebook 08.3)

**Transformation:** norm_comoving(L) = norm(L) / a(L)

**Result:**
- Physical coordinates: CV = 125% (huge variation)
- Comoving coordinates: CV = 12.7% (nearly stationary)
- Reduction: 89.8%

**Interpretation:** Tokens don't move much in comoving space. Almost all variation is due to metric expansion.

### 4. Uniform Expansion (Notebook 08.3)

**Per-token Hubble constants:**
- Mean: H = 0.1150 ± 0.0079 [1/layer]
- CV: 6.85% (highly uniform)
- No position dependence (scatter plot shows no spatial trend)

**Interpretation:** All tokens expand at nearly identical rates. Expansion is a collective, approximately isotropic phenomenon.

---

## The Proposed Model

### Metric Evolution

The causal metric tensor evolves through layers as:

```
M(L) = [a(L) / a(max(L))]² × M(max(L))
```

where:
- **a(L)** is the empirically measured scale factor
- **M(max(L)) = Cov(γ)⁻¹** is the causal metric extracted from the unembedding matrix (Park et al. 2024)
- **M(0) = δᵢⱼ** is assumed to be Euclidean at the embedding layer

**Equivalently, defining M₀ as the "comoving metric":**

```
M₀ = M(max(L)) / a(max(L))²
M(L) = a(L)² × M₀
```

### Causal Norm Under This Model

For a token activation h(L):

```
||h(L)||²_M(L) = h(L)ᵀ M(L) h(L)
                = a(L)² × h(L)ᵀ M₀ h(L)
                = a(L)² × ||h_comoving||²_M₀
```

If h_comoving is approximately constant (tokens stationary in comoving space), then:

```
||h(L)||_M(L) ∝ a(L)
```

which is exactly what we observe.

### Physical Interpretation

1. **Embeddings start in Euclidean space** (M(0) = δᵢⱼ)
2. **Each layer expands the metric** by a factor related to H(L)
3. **Tokens remain approximately stationary** in comoving coordinates
4. **Unembedding measures distance** in the fully expanded metric M(max(L))

The transformer doesn't primarily *move* tokens through semantic space — it **expands the space itself**, stretching all distances proportionally.

---

## Testable Predictions

### Already Confirmed
- ✓ Token norms grow exponentially through layers
- ✓ Expansion is uniform across tokens (CV = 6.85%)
- ✓ Comoving coordinates are stable (89.8% CV reduction)

### Not Yet Tested

**1. M(L) has constant shape**
- **Prediction:** Eigenvectors of M(L) are constant across layers, only eigenvalues scale as a(L)²
- **Test:** Extract or estimate M(L) at multiple layers, compute eigendecomposition
- **Alternative:** M(L) changes shape (anisotropic expansion in different directions)

**2. M(0) is actually Euclidean**
- **Prediction:** Metric at embedding layer is δᵢⱼ (no learned structure)
- **Test:** Compute M(0) from layer-0 activations
- **Alternative:** Embeddings already have non-Euclidean structure

**3. LayerNorm removes expansion**
- **Prediction:** Norms measured *after* LayerNorm should be approximately constant
- **Test:** Measure norms post-LayerNorm at each layer
- **Alternative:** Expansion persists even after LayerNorm (it's fundamental, not artifact)

**4. H(L) is model-invariant**
- **Prediction:** Different models have similar H(L) curves (same epoch structure)
- **Test:** Run 08.1-08.3 on GPT-2, Llama, Gemma, etc.
- **Alternative:** H(L) is model-specific or text-dependent

**5. Steering effectiveness correlates with H(L)**
- **Prediction:** Low-H layers (L=8-15, stable expansion) are better for steering
- **Test:** Extract steering vectors at different layers, measure effectiveness
- **Alternative:** Steering effectiveness depends on other factors (attention, layer position, etc.)

**6. Redshift predicts steering transfer**
- **Prediction:** Steering vector extracted at L₁ and applied at L₂ has effective magnitude V × (1 + z) where z = a(L₂)/a(L₁) - 1
- **Test:** Extract vector at one layer, apply at another, measure actual vs predicted magnitude
- **Alternative:** Steering doesn't scale with redshift

---

## Alternative Explanations Not Ruled Out

### 1. LayerNorm Artifact

**Claim:** Norm growth is just accumulated residual additions. LayerNorm normalizes this away in practice, so the "expansion" is a measurement artifact with no physical significance.

**Counter-evidence needed:** Measure post-LayerNorm norms. If they're constant, this explanation holds. If they still grow, expansion is real.

### 2. Architectural Consequence, Not Semantic

**Claim:** Norm growth is simply N_residual_connections × avg_residual_magnitude. It's mechanical, not meaningful.

**Counter-evidence needed:** Show that H(L) correlates with semantic structure (e.g., attention vs MLP ratio, information content, etc.) rather than just architectural depth.

### 3. M(max(L)) Isn't Special

**Claim:** Cov(γ)⁻¹ is just one possible metric. Using a different metric (Fisher information, learned metric, etc.) would give different results.

**Counter-evidence needed:** Try alternative metrics and show results are consistent, or prove Cov(γ)⁻¹ is unique/optimal.

### 4. Text-Specific or Model-Specific

**Claim:** These results are specific to this one text, this one model, or this random seed. Not universal.

**Counter-evidence needed:** Replicate on:
- Multiple texts (different lengths, topics, languages)
- Multiple models (different architectures, sizes, training datasets)
- Multiple random initializations of the same model

### 5. It's Just Vector Magnitude, Not Metric Expansion

**Claim:** Vectors get bigger through layers. You're calling this "metric expansion" but it's just ||h(L)|| increasing. No evidence the metric itself changes.

**Counter-evidence needed:** Directly measure or estimate M(L) at different layers and show it scales as predicted.

---

## Cosmological Analogies

### Redshift

In cosmology: `1 + z = λ_observed / λ_emitted = a_observed / a_emitted`

For transformers: `1 + z(L₁→L₂) = ||h(L₂)||_M / ||h(L₁)||_M = a(L₂) / a(L₁)`

**Interpretation:** Redshift measures how much the metric has expanded between two layers.

**Application:** Steering vectors are "redshifted" when transferred between layers. A vector designed for layer L₁ will be stretched by (1+z) when applied at layer L₂.

### Hubble Flow

In cosmology: Galaxies recede from each other due to expansion, not because they're moving through space.

For transformers: Token representations diverge through layers because the metric expands, not because they're computing different values.

### Comoving Coordinates

In cosmology: Galaxies are approximately stationary in comoving coordinates. All recession is due to metric expansion.

For transformers: Tokens are approximately stationary in comoving coordinates (CV = 12.7%). Almost all norm growth (89.8%) is metric expansion.

### Cosmic Horizon

In cosmology: Regions beyond the cosmic horizon cannot causally influence each other (expansion faster than light).

For transformers: Is there a "layer horizon" beyond which representations are so redshifted they can't communicate? Could explain why very deep models (100+ layers) don't help much.

### Time Dilation

In cosmology: Processes at high redshift appear time-dilated when observed from z=0.

For transformers: Gradients propagating from early to late layers are "diluted" by expansion factor (1+z). This might explain why training deep transformers requires LayerNorm — without it, gradients redshift into oblivion.

---

## Connection to Previous Results

### Manifold Radius of Curvature (Notebook 06.4c)

We found R(L) grows from 56 logometers (layer 0) to 8227 logometers (layer 35).

**Under the expansion model:**
```
R(L) = a(L) × R₀
```

The manifold "gets flatter" because curvature scales as 1/R², which grows slower than distance (scales as a(L)).

**Check:** R(35)/R(0) = 8227/56 = 147×

But we measured a(35) = 76.9×. These don't match.

**Possible explanations:**
1. Curvature grows independently of expansion (intrinsic vs extrinsic)
2. The steering vector norm growth isn't the same as the scale factor
3. Measurement error or different samples
4. The model is wrong

**Status:** Unresolved discrepancy that needs investigation.

### Steering Vector Norms (Notebook 02)

Layer 35 has the largest Euclidean norm for steering vectors.

**Under the expansion model:**
```
||V(L)||_M = a(L) × ||V_comoving||_M₀
```

If V_comoving is approximately constant (same semantic direction in comoving space), then steering vectors should grow proportionally to a(L).

**Status:** Consistent with observations, but not rigorously tested.

---

## Open Questions

### Theoretical

1. **Why would the model learn to expand from Euclidean to causal?**
   - Is there a loss function or optimization reason this is optimal?
   - Or is it an emergent property of transformer architecture?

2. **Is M(0) = δᵢⱼ actually true?**
   - Do embeddings impose structure from the start?
   - Or do they truly begin Euclidean?

3. **What drives the H(L) curve shape?**
   - Why inflation at L=0?
   - Why deceleration then re-acceleration?
   - Why the Big Crunch at L=34-36?
   - Is this architecturally determined or learned from data?

4. **Is expansion isotropic or anisotropic?**
   - Do all semantic dimensions expand at the same rate?
   - Or do some directions (e.g., high-information features) expand faster?

5. **What's the relationship between expansion and attention/MLP?**
   - Do attention layers cause expansion (tokens spreading out)?
   - Do MLP layers cause compression (processing in place)?
   - Can we decompose H(L) into attention vs MLP contributions?

### Experimental

6. **How do you measure M(L) at intermediate layers?**
   - Extract activations at layer L
   - Compute covariance and invert?
   - Or fit a metric to satisfy some optimality criterion?

7. **Does this generalize to other models?**
   - GPT-2, Llama, Gemma, etc.
   - Do they have similar H(L) curves?
   - Or is this Qwen-specific?

8. **Does this generalize to other texts?**
   - Different lengths, topics, languages
   - Is H(L) text-dependent or universal?

9. **What happens during training?**
   - Does a(L) evolve during training?
   - Does H(L) stabilize or change?
   - Is there a relationship between training loss and expansion rate?

10. **Can we use this for practical applications?**
    - Better steering (accounting for redshift)
    - Better layer selection (target low-H regions)
    - Better interpretability (comoving coordinates as "true" representation)

---

## Experimental Roadmap

### Phase 1: Validation (Minimum Viable Test)

**Goal:** Confirm the core observations hold across variations

**Experiments:**
1. **Multi-text test:** Run 08.1-08.3 on 5-10 different texts
   - Does a(L) curve stay similar?
   - Is H₀ approximately constant?
   - Do per-token Hubble constants remain uniform?

2. **Post-LayerNorm test:** Measure norms after LayerNorm
   - Do they stay constant or still grow?
   - If constant: expansion is artifact
   - If growing: expansion is fundamental

3. **Multi-model test:** Run on GPT-2 (different architecture)
   - Compare H(L) curves
   - Is the multi-epoch structure universal?

### Phase 2: Mechanism (Understanding Why)

**Goal:** Test the proposed M(L) = a(L)² × M(max(L)) model

**Experiments:**
4. **Direct M(L) estimation:** Compute metric at layers 0, 18, 35
   - Method TBD (covariance inversion? fitted metric? other?)
   - Check if eigenvectors are constant
   - Check if eigenvalues scale as a(L)²

5. **Isotropy test:** Eigenvalue ratio analysis
   - Do all directions expand at same rate?
   - Plot eigenvalue spectrum at different layers

6. **Attention vs MLP decomposition:**
   - Separate attention and MLP contributions to norm growth
   - Does H(L) correlate with attention/MLP ratio?

### Phase 3: Applications (Using The Model)

**Goal:** Leverage the expansion model for practical improvements

**Experiments:**
7. **Redshift-corrected steering:**
   - Extract vector at L₁, apply at L₂ with redshift correction
   - Test if correction improves steering effectiveness

8. **Optimal layer selection:**
   - Predict that low-H layers are better for steering
   - Empirically test steering at different H(L) values

9. **Comoving coordinate training:**
   - Train model with loss in comoving coordinates (divide by a(L))
   - Does this improve stability, generalization, or convergence?

---

## References

**Park et al. (2024):** "Linearity of Relation Decoding in Transformer Language Models"
- Defined causal metric tensor M = Cov(γ)⁻¹ from unembedding matrix
- We extend this to layer-dependent M(L)

**FLRW Metric (cosmology):**
- Friedmann-Lemaître-Robertson-Walker metric for expanding universe
- Provides mathematical framework for scale factor a(t) and Hubble parameter H(t)

**Project Azimuth notebooks:**
- 08.1: Extract token activations
- 08.2: Analyze norm evolution
- 08.3: Fit FLRW expansion model
- 06.4c: Manifold radius in logometers

---

## Status: Hypothesis

This is a **phenomenological model** that fits our observations elegantly. However:

- ✗ Not yet validated on multiple texts/models
- ✗ Not yet proven that M(L) actually scales this way
- ✗ Not yet ruled out simpler alternative explanations
- ✗ Not yet connected to training dynamics or loss function

**Next step:** Phase 1 validation experiments (multi-text, post-LayerNorm, multi-model)

**Promotion to findings/:** Requires passing Phase 1 + directly measuring M(L) in Phase 2

---

**Last updated:** October 29, 2025
**Authors:** Jeffery Harrell, Alpha
