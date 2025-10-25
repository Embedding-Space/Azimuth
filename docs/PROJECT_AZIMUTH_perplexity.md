# Project Azimuth: Measuring Self-Perplexity During Inference

## Motivation

When steering language models with intervention vectors, we need to know: **"How lost is the model?"**

Two key metrics tell the story:
- **Grade Level**: *What* is the model saying? (semantic content)
- **Self-Perplexity**: *How confident* is the model about what it's saying? (epistemic state)

Traditional perplexity measurement requires re-running the model token-by-token through generated text—expensive! But we can get it **for free** during generation by capturing probabilities that are already being computed.

## What is Self-Perplexity?

**Self-perplexity** is: "How surprised is the model by its own generation?"

Given generated tokens [t₁, t₂, ..., tₙ], perplexity is:

```
PPL = exp(-1/n · Σ log P(tᵢ | t₁, ..., tᵢ₋₁))
```

**Low perplexity (≈1-20)**: Model is confident, distributions are peaked
- "The most likely token is very likely"
- On-manifold behavior

**High perplexity (100+)**: Model is confused, distributions are flat  
- "Everything seems equally unlikely"
- Off-manifold behavior

### Key Insight: Greedy Decoding Still Has Perplexity

Even with temperature=0 (greedy), perplexity can be high!

**On-manifold** (α=0):
```
P("the") = 0.7  ← greedy picks this
P("a") = 0.15
P("an") = 0.10
... everything else < 0.01
```
log P("the") = -0.36 → PPL ≈ 1.4 (confident)

**Off-manifold** (α=50):
```
P("the") = 0.12  ← greedy still picks highest
P("a") = 0.11
P("quantum") = 0.10
P("purple") = 0.09
... nearly uniform distribution
```
log P("the") = -2.12 → PPL ≈ 8.3 (confused)

**High perplexity with greedy means**: "Even my best guess is implausible."

## The Trick: Capture Scores During Generation

The Hugging Face `transformers` library already computes logits at each generation step. We just need to save them!

### Basic Implementation

```python
outputs = model.generate(
    input_ids,
    max_length=100,
    do_sample=False,  # Greedy decoding
    return_dict_in_generate=True,
    output_scores=True  # ← THE MAGIC FLAG
)

generated_tokens = outputs.sequences[0]
logits_per_step = outputs.scores  # List of [vocab_size] tensors
```

Now compute perplexity:

```python
log_probs = []
for i, token_id in enumerate(generated_tokens[prompt_length:]):
    logits = logits_per_step[i]  # [vocab_size]
    probs = torch.softmax(logits, dim=-1)
    log_probs.append(torch.log(probs[token_id]))

avg_log_prob = torch.stack(log_probs).mean()
perplexity = torch.exp(-avg_log_prob)
```

**Cost**: Zero extra forward passes! Just bookkeeping.

## Complete Implementation for Project Azimuth

```python
# In azimuth/measurement.py

def generate_with_self_perplexity(
    model, 
    prompt, 
    steering_vector=None, 
    alpha=0, 
    max_length=100,
    layer=27
):
    """
    Generate text and compute self-perplexity in one pass.
    
    Args:
        model: The language model
        prompt: Input text (string)
        steering_vector: Optional steering vector to apply
        alpha: Steering coefficient
        max_length: Maximum tokens to generate
        layer: Which layer to apply steering
        
    Returns:
        dict with keys:
            - text: Generated text (string)
            - perplexity: Self-perplexity (float)
            - tokens: Generated token IDs
            - log_probs: Per-token log probabilities (for analysis)
    """
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
    
    # Apply steering if provided
    with steering_context(model, steering_vector, alpha, layer):
        outputs = model.generate(
            input_ids,
            max_length=max_length,
            do_sample=False,  # Greedy decoding for determinism
            return_dict_in_generate=True,
            output_scores=True  # Capture logits
        )
    
    generated_tokens = outputs.sequences[0]
    prompt_length = len(input_ids[0])
    generated_text = tokenizer.decode(generated_tokens[prompt_length:])
    
    # Compute self-perplexity from saved scores
    log_probs = []
    for i, token_id in enumerate(generated_tokens[prompt_length:]):
        logits = outputs.scores[i]
        log_prob = torch.log_softmax(logits, dim=-1)[token_id]
        log_probs.append(log_prob)
    
    perplexity = torch.exp(-torch.stack(log_probs).mean())
    
    return {
        'text': generated_text,
        'perplexity': perplexity.item(),
        'tokens': generated_tokens[prompt_length:],
        'log_probs': [lp.item() for lp in log_probs]
    }
```

## Memory Considerations

`output_scores=True` keeps all logits: `vocab_size × num_generated_tokens × 4 bytes`

For 100 tokens × 128k vocab × 4 bytes ≈ **50 MB per generation**

For a 20×20 grid: 400 generations × 50 MB = **20 GB**

### Solutions

**Option 1: Compute and discard** (recommended)
```python
# Only keep final perplexity value
result = generate_with_self_perplexity(...)
# Returns: {'text': ..., 'perplexity': 42.3}
# Scores discarded after computation
```

**Option 2: Efficient storage**
```python
# Only save log prob of chosen token (tiny!)
log_probs = []  # Just n floats, not n × vocab_size
for i, token_id in enumerate(generated_tokens[prompt_length:]):
    logits = outputs.scores[i]
    log_prob = torch.log_softmax(logits, dim=-1)[token_id]
    log_probs.append(log_prob.item())  # Single float
    # Discard full logits tensor
```

**Option 3: Detailed analysis mode**
```python
# Keep everything for deep inspection (use sparingly)
result = generate_with_full_diagnostics(...)
# Returns full logits, token surprises, entropy per step, etc.
```

## Application: 2D Steering Sweeps

Measure both grade level AND perplexity across (α, β) grid:

```python
# In azimuth/experiments/sweep_2d.py

results = []
for alpha in np.linspace(-10, 10, 20):
    for beta in np.linspace(-10, 10, 20):
        steering = alpha * v_complexity + beta * v_perp
        
        result = generate_with_self_perplexity(
            model, 
            prompt="Explain photosynthesis: ",
            steering_vector=steering,
            alpha=1.0,  # Already scaled into 'steering'
            max_length=100
        )
        
        results.append({
            'alpha': alpha,
            'beta': beta,
            'grade_level': compute_grade_level(result['text']),
            'perplexity': result['perplexity'],
            'text': result['text']
        })

# Create dual heatmaps
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Semantic content
ax1.imshow(grade_level_grid, extent=[-10, 10, -10, 10])
ax1.set_title('Grade Level (What is said)')

# Model confidence  
ax2.imshow(perplexity_grid, extent=[-10, 10, -10, 10])
ax2.set_title('Self-Perplexity (How lost is model)')
```

## Interpretation Guide

### Scenario A: Low GL, Low PPL
```
Grade Level: 8.0
Perplexity: 15.0
```
**Interpretation**: Model is on-manifold, confidently generating simple text.

### Scenario B: High GL, Low PPL  
```
Grade Level: 18.0
Perplexity: 20.0
```
**Interpretation**: Model is on-manifold, confidently generating complex text.

### Scenario C: Moderate GL, High PPL
```
Grade Level: 12.0  
Perplexity: 150.0
```
**Interpretation**: Text looks okay semantically, but model is confused. Probably near manifold boundary.

### Scenario D: Undefined GL, Very High PPL
```
Grade Level: NaN (gibberish)
Perplexity: 5000.0
```
**Interpretation**: Completely off-manifold. Model is generating tokens it finds implausible. Semantic breakdown.

## Advanced: Token-Level Surprise

Track which specific tokens are surprising:

```python
def analyze_token_surprises(result):
    """Identify which tokens had unexpectedly low probability."""
    tokens = result['tokens']
    log_probs = result['log_probs']
    
    surprises = [-lp for lp in log_probs]  # Negative log prob = surprise
    
    for token, surprise in zip(tokens, surprises):
        token_str = tokenizer.decode([token])
        print(f"{token_str:20s} surprise={surprise:.2f}")
```

Example output:
```
The                  0.3  ← normal
cat                  0.5  ← normal
sat                  0.4  ← normal
on                   0.3  ← normal
the                  0.4  ← normal
quantum              6.9  ← VERY SURPRISING
fluctuation          7.2  ← VERY SURPRISING
```

This pinpoints where the model "loses confidence" in its own generation.

## Experimental Predictions

### Hypothesis: Semantic Archipelago

**Prediction**: Perplexity map will show:
- Low-PPL "islands" around stable attractors (on-manifold regions)
- High-PPL "oceans" between islands (off-manifold voids)  
- Sharp transitions at boundaries

**Test**: 2D sweep with fine resolution (0.5 unit spacing)

### Hypothesis: Discontinuity = Manifold Boundary

**Prediction**: The grade-level discontinuity at α≈4.4 corresponds to crossing a perplexity threshold.

**Test**: 1D sweep along V_complexity, measure both metrics:
```python
for alpha in np.linspace(0, 10, 100):
    result = generate_with_self_perplexity(model, prompt, v_complexity, alpha)
    plot_point(alpha, result['grade_level'], result['perplexity'])
```

Expected: Sharp PPL increase coinciding with GL discontinuity.

### Hypothesis: Multiple Stable Regions

**Prediction**: Pushing α very far (α=20, 50, 100) might find OTHER low-PPL regions—alternative stable attractors.

**Test**: Extreme steering with perplexity monitoring:
```python
for alpha in [1, 5, 10, 20, 50, 100]:
    result = generate_with_self_perplexity(model, prompt, v_complexity, alpha)
    print(f"α={alpha}: GL={result['grade_level']:.1f}, PPL={result['perplexity']:.1f}")
```

If PPL increases monotonically: single manifold, model gets progressively lost.
If PPL has local minima: multiple stable regions ("semantic galaxies").

## Cost Analysis

### Traditional Approach (Re-evaluation)
- Generate text: 1 forward pass per token × n tokens = **n passes**
- Compute perplexity: n contexts × 1 pass each = **n passes**  
- **Total: 2n forward passes**

### Azimuth Approach (Free Perplexity)
- Generate text with `output_scores=True`: **n passes**
- Compute perplexity from saved scores: **0 passes** (just arithmetic)
- **Total: n forward passes**

**Savings**: 50% reduction in compute!

For 400 grid points × 100 tokens:
- Traditional: 8,000,000 forward passes
- Azimuth: 4,000,000 forward passes
- **Time saved**: Potentially hours on GPU

## Implementation Checklist for Project Azimuth

- [ ] Add `output_scores=True` to all generation calls
- [ ] Implement `generate_with_self_perplexity()` in `azimuth/measurement.py`
- [ ] Add perplexity column to all sweep result CSVs
- [ ] Create dual heatmap visualization (GL + PPL)
- [ ] Add token-level surprise analysis for debugging
- [ ] Test memory usage on 20×20 grid
- [ ] Document perplexity interpretation guide
- [ ] Add perplexity threshold detection for manifold boundaries

## Summary

**Key Insight**: Self-perplexity comes free during generation by capturing probabilities that are already computed.

**Why It Matters**: Perplexity tells us when we've left the manifold—when the model is generating text it finds implausible. This is invisible if we only measure semantic properties like grade level.

**The Promise**: Dual metrics (semantics + confidence) give us a complete picture of where models behave naturally vs where they struggle, enabling principled exploration of activation space geometry.

---

*Project Azimuth: Mapping the bearings of semantic space, one steering vector at a time.*

*Pronunciation: AZ-ə-muth (first syllable stress)*
