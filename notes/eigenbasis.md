# Working in the Eigenbasis of the Causal Metric

**Date:** October 26, 2025
**Status:** Exploratory idea - not yet tested

## The Idea

The causal metric tensor $M = \text{Cov}(\gamma)^{-1}$ defines a natural geometry on semantic space. Currently we work in the "gamma basis" (the native coordinate system of the model's `lm_head.weight` matrix), where $M$ is a full 2560×2560 matrix with complex structure.

**Proposal:** Transform to the eigenbasis of $M$, where the metric becomes diagonal. This might simplify geometry, reveal interpretable semantic dimensions, and provide better tools for analyzing steering vectors.

## Mathematical Background

### The Metric Tensor

The causal metric tensor measures distances based on the model's probability distribution:

$$M = \text{Cov}(\gamma)^{-1}$$

where $\gamma \in \mathbb{R}^{V \times d}$ is the unembedding matrix (vocab size $V$ = 151,936, hidden dimension $d$ = 2560).

**Causal distance** between two token embeddings $v_1, v_2$:

$$d_M(v_1, v_2) = \sqrt{(v_1 - v_2)^T M (v_1 - v_2)}$$

**Causal norm** of a vector $v$:

$$||v||_M = \sqrt{v^T M v}$$

### Eigendecomposition

Since $M$ is symmetric and positive definite, it has a complete eigendecomposition:

$$M = Q \Lambda Q^T$$

where:
- $Q \in \mathbb{R}^{d \times d}$ is orthogonal matrix of eigenvectors (columns)
- $\Lambda = \text{diag}(\lambda_1, \lambda_2, \ldots, \lambda_d)$ is diagonal matrix of eigenvalues
- Eigenvalues sorted: $\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_d > 0$

**Interpretation:**
- **Eigenvectors:** "Natural" orthogonal axes for the causal metric
- **Eigenvalues:** Importance weights for each axis

### Change of Basis

**Transform from gamma basis to eigenbasis:**

$$v_{\text{eigen}} = Q^T v_{\text{gamma}}$$

**Transform back:**

$$v_{\text{gamma}} = Q \, v_{\text{eigen}}$$

**Key property:** The geometry is preserved - distances and norms computed in either basis give identical results.

## Why This Might Be Useful

### 1. Simpler Distance Computation

**Gamma basis (current):**

$$||v||_M^2 = v^T M v = \sum_{i,j} v_i M_{ij} v_j \quad \text{(O(d²) operations)}$$

**Eigenbasis:**

$$||v||_M^2 = v^T \Lambda v = \sum_{i=1}^d \lambda_i v_i^2 \quad \text{(O(d) operations)}$$

No matrix multiplication needed - just a weighted sum of squared coordinates.

### 2. Clear Importance Hierarchy

Each eigendirection has an explicit importance weight $\lambda_i$:

- **Large $\lambda$:** This dimension is heavily weighted in semantic distance
- **Small $\lambda$:** This dimension barely matters

**Example from our data:**
- $\lambda_1 = 9.42 \times 10^4$ (most important direction)
- $\lambda_{\text{median}} = 2.50 \times 10^3$
- $\lambda_{2560} = 9.54 \times 10^1$ (least important)

The first eigendirection is ~1000× more important than the last.

### 3. Natural Dimensionality Reduction

The participation ratio tells us only ~1333 of 2560 dimensions are "active." In eigenbasis, we can:

1. Keep top $k \approx 1333$ eigenvectors (high $\lambda$)
2. Drop bottom $\sim 1227$ eigenvectors (low $\lambda$, noise)
3. Work in effective 1333D space with minimal information loss

This is like PCA, but using the **causal metric** instead of Euclidean variance.

### 4. Detecting Off-Manifold Points

Tokens on the semantic manifold should:
- Have most weight in high-$\lambda$ dimensions
- Have negligible weight in low-$\lambda$ dimensions

**Off-manifold indicators:**
- Significant components in low-$\lambda$ directions
- Unusual causal norm (not $\approx 54$ units)

### 5. Analyzing Steering Vectors

When we extract a steering vector (e.g., complexity), we can:

1. Transform to eigenbasis: $v_{\text{steer,eigen}} = Q^T v_{\text{steer}}$
2. See which semantic dimensions it moves along
3. Check if it stays in the high-$\lambda$ subspace (on-manifold steering)
4. Compare different extraction methods by their eigenbasis decomposition

## Open Questions

### Are Eigendirections Interpretable?

**Hypothesis:** The eigenvectors might correspond to interpretable semantic dimensions.

**Why this might be true:**
- They're the "natural" axes defined by the model's probability distribution
- They're sorted by importance - the model "cares most" about eigenvector 1
- Park et al. (2024) found linear structure in semantic space
- They're orthogonal - each is an independent dimension

**Why this might be false:**
- Eigendecomposition maximizes variance, not semantic interpretability
- Could be arbitrary rotations of interpretable features (like PCA vs ICA)
- High-$\lambda$ directions might be "structural" (e.g., special tokens) not "semantic"

**How to test:**
1. Find tokens with extreme (positive/negative) components in each eigendirection
2. Look for semantic clustering (e.g., does eigenvector 5 separate abstract/concrete?)
3. Probe with known semantic axes (king-queen, he-she, happy-sad)
4. Check if steering vectors align with particular eigendirections

### Do Tokens Really Live in ~1333D?

The participation ratio suggests effective dimensionality ≈ 1333. In eigenbasis, we could:

1. Project tokens onto top-$k$ eigenvectors for varying $k$
2. Measure information loss (reconstruction error)
3. Find the "knee" where adding more dimensions gives diminishing returns
4. Test if $k \approx 1333$ is sufficient for preserving semantic structure

### Does Steering Quality Correlate with Eigenbasis Structure?

**Prediction:** Good steering vectors should:
- Have most weight in high-$\lambda$ dimensions (semantically important)
- Stay within the effective 1333D subspace (on-manifold)
- Not leak into low-$\lambda$ dimensions (off-manifold)

**Test:** Compare steering vectors extracted by different methods (L2 vs causal norm, different layers) in eigenbasis coordinates.

## Practical Considerations

### Computational Cost

**One-time cost:**
- Eigendecomposition of 2560×2560 matrix: ~10 seconds
- Already computed in notebook 03

**Per-token transform:**
- $v_{\text{eigen}} = Q^T v_{\text{gamma}}$: O($d^2$) = ~6M operations
- Fast on modern hardware, can batch

**Storage:**
- $Q$ matrix: 2560×2560 float32 = 26 MB (negligible)
- Can store pre-transformed tokens if working entirely in eigenbasis

### When to Use Eigenbasis

**Good use cases:**
- Analyzing semantic structure (which directions matter?)
- Comparing steering vectors (do they move in similar directions?)
- Dimensionality reduction (compress to effective dimensions)
- Detecting off-manifold behavior (check low-$\lambda$ leakage)

**When to stay in gamma basis:**
- Need to interface with model directly (model expects gamma basis)
- One-off distance calculations (transform overhead not worth it)
- Interpretability matters more than geometry (gamma basis might be more interpretable)

## Connection to Existing Work

### Relation to PCA

Standard PCA finds eigenvectors of the **Euclidean covariance** matrix $\text{Cov}(\gamma)$.

Our eigenbasis comes from $M = \text{Cov}(\gamma)^{-1}$, the **inverse** covariance.

**Key difference:**
- **PCA:** Directions of maximum variance in token embeddings
- **Eigenbasis of M:** Directions of maximum importance for probability distribution

These are mathematically dual - high variance in $\text{Cov}(\gamma)$ corresponds to low importance (small $\lambda$) in $M$.

### Relation to Park et al. (2024)

Park et al. introduced the causal metric and showed linear structure in relation representations. They worked primarily in the gamma basis.

**Our contribution (if this works):** Showing that the **eigenbasis** of the causal metric might be the natural coordinate system for semantic analysis.

## Next Steps

**Proposed notebook: `05_eigenbasis_analysis.ipynb`**

1. **Transform tokens to eigenbasis**
   - Precompute $Q$ (already have from 03)
   - Transform sample of tokens: $v_{\text{eigen}} = Q^T v_{\text{gamma}}$

2. **Explore eigendirection semantics**
   - For each of top 20 eigenvectors:
     - Find tokens with largest positive/negative components
     - Display token strings, look for patterns
   - Test hypothesis: are eigendirections interpretable?

3. **Dimensionality analysis**
   - Project tokens onto top-$k$ eigenvectors for $k \in [100, 500, 1000, 1333, 2000, 2560]$
   - Measure reconstruction error vs $k$
   - Validate effective dimensionality ≈ 1333

4. **Steering vector analysis**
   - Load complexity steering vector (from notebook 02)
   - Transform to eigenbasis
   - Plot weight distribution across eigendirections
   - Check: does it stay in high-$\lambda$ subspace?

5. **On-manifold vs off-manifold**
   - Sample random points in eigenbasis
   - Some constrained to high-$\lambda$ subspace (on-manifold)
   - Some with weight in low-$\lambda$ dimensions (off-manifold)
   - Transform to gamma basis, run through model
   - Measure: perplexity, generation quality
   - Test: does off-manifold → higher perplexity?

## Summary

The eigenbasis of the causal metric tensor $M$ provides a natural coordinate system where:
- Distances are simple weighted sums (diagonal metric)
- Dimensions are explicitly ranked by importance (eigenvalues)
- Geometric structure might be clearer and more interpretable

**Open question:** Are the eigendirections semantically meaningful?

**Hypothesis:** Working in eigenbasis will simplify analysis and potentially reveal interpretable semantic dimensions that are obscured in the gamma basis.

**Status:** Idea proposed, not yet tested. Worth exploring in next notebook.

---

## Mathematical Appendix

### Verification: Distance is Preserved

Given two vectors $v_1, v_2$ in gamma basis and their transforms $u_1 = Q^T v_1$, $u_2 = Q^T v_2$ in eigenbasis:

$$
\begin{align}
d_M(v_1, v_2)^2 &= (v_1 - v_2)^T M (v_1 - v_2) \\
&= (v_1 - v_2)^T Q \Lambda Q^T (v_1 - v_2) \\
&= (Q^T(v_1 - v_2))^T \Lambda (Q^T(v_1 - v_2)) \\
&= (u_1 - u_2)^T \Lambda (u_1 - u_2) \\
&= \sum_{i=1}^d \lambda_i (u_{1,i} - u_{2,i})^2
\end{align}
$$

Same distance, computed in either basis. ✓

### Eigenvalue Statistics (Qwen3-4B)

From our measurement:
```
Eigenvalues of M:
  λ₁ (max):     9.42 × 10⁴
  λ₁₂₈₀ (median): 2.50 × 10³
  λ₂₅₆₀ (min):    9.54 × 10¹

Effective dimensions: 1333 / 2560 (52%)
```

The eigenvalue spectrum spans ~1000× range, indicating strong anisotropy - some directions matter far more than others.
