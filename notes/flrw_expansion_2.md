# Final Layer Metric Contraction Hypothesis

## The Proposal

Layer 35's metric contraction serves to compress the space between the final hidden state and the output token manifold, making unembedding more efficient.

## Mechanism

1. **Layers 0-34**: Metric expands with scale factor a(L), tokens spread out, semantic space grows
2. **Layer 35**: Metric contracts (a(35) < a(34)), bringing everything closer together  
3. **Unembedding**: Takes the compressed h(35) and computes dot products with γ (152k token vectors)

The contraction makes the dot products **more discriminating** - by bringing h closer to the correct token neighborhood, you increase the gap between the best match and the alternatives.

## Testable Predictions

### 1. Token Density Through Layers
Compute mean nearest-neighbor distance in causal metric at each layer.

**Prediction:**
- Distance should decrease from L=0 to some maximum
- Distance increases (space expands) through L=34
- Distance drops sharply at L=35 (compression)

**Method:**
- Sample N tokens from vocabulary (e.g., 1000-5000)
- For each layer L, compute token embeddings/representations at that layer
- Calculate mean k-nearest-neighbor distance using causal metric
- Plot across all layers

### 2. Angular Alignment to Top-K Tokens
Measure cosine similarity between h(L) and its top-k predicted tokens.

**Prediction:**
- Angular alignment should increase at L=35 (vectors becoming more parallel)
- h(35) should be more aligned with correct output tokens than h(34)

**Method:**
- For representative prompts, capture h(L) at each layer
- Compute cosine similarity between h(L) and top-k token vectors from γ
- Compare mean alignment at L=34 vs L=35
- Test if alignment increase is statistically significant

### 3. Logit Sharpness / Output Entropy
Measure entropy of output distribution at different layers.

**Prediction:**
- Metric compression → sharper predictions
- Entropy should be lower when measuring at L=35 vs L=34
- The model should be "more confident" after the compression

**Method:**
- Generate text with output from L=34 vs L=35
- Compute entropy of softmax distribution over vocabulary
- Compare distributions: L=35 should show lower entropy
- Alternatively: measure "effective vocabulary size" (perplexity)

## Interpretation

The Big Crunch isn't just an artifact - it's **functional geometry**. The model expands semantic space through layers 0-34 to do its thinking, perform computations, and build rich representations. Then it compresses the space at layer 35 to facilitate efficient mapping to discrete output tokens.

This is analogous to:
- **Expansion phase**: Working memory, semantic processing, context integration
- **Contraction phase**: Decision formation, output preparation, "zooming in" on the answer

The final layer serves as a geometric adapter between the high-dimensional semantic workspace and the discrete token vocabulary.

## Status

**Hypothesis**: Proposed Oct 30, 2025  
**Evidence**: Observational (metric scaling behavior from FLRW analysis)  
**Validation**: Awaiting experimental tests