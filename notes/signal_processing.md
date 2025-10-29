# Signal Processing Analysis of Multi-Layer Steering

**Date:** October 29, 2025
**Context:** Finding 03 - Layer Selection via SNR

---

## The Core Idea

When you steer a model by adding `α·V` to activations at different layers, you can treat the responses from all 36 layers as a **multiband signal** - like having 36 different sensors measuring the same underlying phenomenon from different perspectives.

This opens up signal processing techniques to understand steering behavior in ways that single-layer analysis can't reveal.

---

## The Method

### 1. High-Resolution Whole-Model Sweep

**Setup:**
- Steer at **every layer** (0-35 for Qwen3-4B)
- Sweep steering coefficient α ∈ [-10, 10] with high resolution (~0.02 spacing)
- Generate multiple samples at each (layer, α) point
- Measure output metric (e.g., Flesch-Kincaid grade level)

**Result:** A dataset with dimensions [36 layers × 500 α values × N samples] = ~18,000 datapoints

**Cost:** ~$0.40 for complete characterization (one-time per model/vector pair)

### 2. Extract Consensus Signal

**Definition:** The consensus signal is the **mean response across all layers** at each α:

```python
consensus_signal(α) = mean over layers of response(layer, α)
```

**Interpretation:** This represents the "typical" or "expected" steering behavior - what the model *as a whole* does when you steer by amount α, regardless of which layer you intervene at.

### 3. Compute Inter-Layer Variance

**Definition:** At each α, compute the variance across layer responses:

```python
variance(α) = var over layers of response(layer, α)
```

**Interpretation:**
- **Low variance** → layers agree on what α means (on-manifold, KALM)
- **High variance** → layers wildly disagree (off-manifold, PANIK)

Variance is a probe for manifold membership.

### 4. Compute Variance Derivative

**Definition:** The rate of change of variance with respect to steering:

```python
d(variance)/dα ≈ gradient(variance, α)
```

Use `np.gradient()` for centered finite differences.

**Interpretation:**
- **Low derivative** → variance is stable, steering is predictable
- **High derivative** → rapidly leaving/entering stable region

The "flat floor" in the derivative plot identifies the steerable range.

### 5. Bin and Smooth

**Purpose:** Raw derivative is noisy. Bin into uniform buckets (bin_size = 0.5 to 1.0) and average:

```python
for each bin [α_i, α_i + bin_size):
    binned_derivative[i] = mean(|d(variance)/dα| in bin)
```

This reveals the underlying structure without point-to-point noise.

---

## The Discovery: Exponential-Quadratic Relationship

### Observation

When you plot `binned_derivative` on a **logarithmic Y-axis**, it looks like a **parabola**.

### Mathematical Implication

If log-plot shows a parabola, then:

```
log(|d(variance)/dα|) ≈ k·α² + b·α + c
```

Taking the exponential of both sides:

```
|d(variance)/dα| ≈ exp(k·α²)  (ignoring linear/constant terms)
```

This is a **Gaussian-exponential** - low in the middle, shooting up quadratically-exponentially on both sides.

### Integration to Get Variance

While `∫exp(α²)dα` has no closed form (it's related to the error function), we can numerically approximate:

```python
variance(α) ≈ cumulative_trapezoid(exp(k·α²), α)
```

For discrete data, we already have `variance(α)` measured directly at each α point, so we can just plot it.

**Key insight:** The derivative going as `exp(α²)` means variance grows **super-exponentially** - faster than any polynomial.

---

## Physical Interpretation: Manifold Geometry

### The Manifold Picture

**Hypothesis:** Model representations live on a lower-dimensional manifold embedded in activation space.

**Steering as tangent motion:** Adding α·V moves you along a tangent direction to the manifold at the model's natural operating point (α=0).

**Variance as distance-from-manifold probe:**
- On-manifold → layers agree on representation → low variance
- Off-manifold → layers project ambiguously → high variance

### The exp(α²) Signature

The relationship `variance ∝ exp(k·α²)` is the **signature of Gaussian curvature**.

**Geometric interpretation:**
- α is linear displacement along tangent vector
- α² is (approximate) squared distance from manifold
- exp(α²) is exponential growth of ambiguity as you leave the surface

**Analogy:** Standing on a hilltop (manifold). Close by, everyone sees where you are. Far away in the valley, you're equally close to multiple hills - ambiguous position → high variance.

The quadratic exponent **measures the local curvature** of the manifold in the direction perpendicular to the steering vector.

---

## Implications for Linear Representation Hypothesis

### LRH Claims

The Linear Representation Hypothesis says concepts are represented as **linear directions** in activation space. Steering by α·V assumes:

```
concept_strength ∝ α  (linear relationship)
```

### What We Found

Linear relationship holds **only locally** - in the KALM region where variance is low and roughly constant.

Outside that region:
- Variance grows as exp(α²)
- Layers disagree exponentially
- Linear approximation breaks down catastrophically

### Refined Statement

**"Linear Representation Hypothesis (Local Form)":**
Concepts are represented as linear directions in activation space **within a finite radius of the model's natural operating point**. The radius of validity is determined by where inter-layer variance begins exponential growth.

**Implication:** LRH describes the **tangent space** at α=0, not the global geometry. It's a first-order Taylor approximation, valid only in the KALM region.

The exponential breakdown quantifies exactly **how far you can steer** before leaving the linear regime.

---

## The PANIK-KALM-PANIK Framework

### Three Regimes

1. **PANIK** (α < -3): High variance, exponentially growing, off-manifold
2. **KALM** (α ∈ [-3, +6]): Low variance, flat derivative, on/near manifold, linear regime valid
3. **PANIK** (α > +6): High variance, exponentially growing, off-manifold

### Identification Algorithm

```python
# Compute derivative of variance
dvar_dalpha = np.gradient(variance_by_alpha, alpha_values)

# Bin and smooth
binned = bin_and_average(abs(dvar_dalpha), bin_size=1.0)

# Plot on log scale to see parabolic structure
plt.yscale('log')
plt.bar(bin_centers, binned)

# Identify KALM region: where binned derivative is below threshold
threshold = np.percentile(binned, 25)  # or use visual inspection
kalm_mask = binned < threshold
```

### Practical Use

**For steering applications:** Stay within the KALM region to ensure:
- Predictable, repeatable behavior
- Linear relationship between α and effect
- Inter-layer agreement (low model confusion)
- On-manifold operation (semantically meaningful)

**For research:** The boundaries of the KALM region tell you:
- The size of the "trust region" for linear interventions
- The local curvature of the representation manifold
- How robust the steering vector is to scaling

---

## Open Questions

1. **Universality:** Does exp(α²) hold for other steering vectors? Other models? Other metrics besides grade level?

2. **Asymmetry:** The KALM region appears slightly asymmetric ([-3, +6] not centered at zero). What does this mean about the model's natural bias?

3. **Cubic connection:** Finding 02 showed LayerNorm has a cubic Taylor expansion. How does that relate to the quadratic-exponential variance growth? Are we seeing two different phenomena or the same thing from different angles?

4. **Multi-dimensional steering:** What happens when you steer along two vectors simultaneously? Does variance grow as exp(α₁² + α₂²) (spherical) or something else?

5. **Layer-specific curvature:** Different layers might have different local curvatures. Can we extract per-layer manifold geometry from this data?

---

## Summary

By treating multi-layer steering data as a signal processing problem, we discovered that **inter-layer variance grows exponentially with the square of steering magnitude**: `variance ∝ exp(k·α²)`.

This isn't just an empirical observation - it's the **geometric signature** of steering away from a curved manifold in activation space. The exponential breakdown quantifies the radius of validity for the Linear Representation Hypothesis and defines the "safe operating region" (KALM) for steering interventions.

**Bottom line:** Linear steering works, but only locally. Step outside the KALM region and you fall off the manifold exponentially fast.

---

**Next steps:**
- Test on other steering vectors and models
- Fit the quadratic exponent k to quantify manifold curvature
- Investigate connection to LayerNorm cubic nonlinearity
- Explore multi-dimensional steering geometry
