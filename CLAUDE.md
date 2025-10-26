# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

# Project Azimuth

**Pronunciation:** AZ-ə-muth (first syllable stress)

**Tagline:** *Mapping the bearings of semantic space, one steering vector at a time.*

## What This Is

Project Azimuth is a "serious-for-fun" research project exploring activation steering in large language models. We're measuring **both** semantic content (what the model says) and epistemic state (how confident it is) during steering experiments.

### Core Insight

Self-perplexity comes **free** during generation by capturing probabilities that are already computed. Just add `output_scores=True` to your generation call — no extra forward passes needed. This lets us map the "semantic archipelago": low-perplexity islands where models are confident, high-perplexity oceans where they're confused, and sharp boundaries where we fall off the manifold.

### Goals

1. Measure both grade level (semantic content) AND perplexity (model confidence) across steering parameter sweeps
2. Map manifold boundaries in activation space
3. Test hypotheses about steering vector extraction methods (semantic vs empirical)
4. Understand when and how models "get lost" during intervention
5. **Explore geometric structure of semantic space** under the causal metric tensor (Park et al. 2024)

## Project Structure

```
Azimuth/
├── azimuth/              # Python package (shared tooling)
│   ├── __init__.py       # Package initialization
│   └── config.py         # Configuration constants
├── notebooks/            # Jupyter notebooks (experiments)
│   ├── 01_datasets.ipynb                # Dataset collection and quality analysis
│   ├── 02_vector_extraction.ipynb       # Steering vector extraction
│   ├── 03_causal_metric_tensor.ipynb    # Extract causal metric tensor M
│   ├── 04.1_metric_properties.ipynb     # Effective dimensionality, non-Euclidean distance, token cloud extent
│   ├── 04.2_forman_ricci_curvature.ipynb # Discrete curvature estimation
│   ├── 04.3_community_detection.ipynb   # Semantic clustering via Louvain
│   └── 04.4_umap_visualization.ipynb    # 2D/3D visualizations of semantic space
├── data/                 # Datasets and extracted vectors
│   ├── onestop_*.{csv,json}  # OneStopEnglish dataset
│   ├── wikipedia_*.{csv,json} # Wikipedia dataset
│   └── vectors/              # Extracted steering vectors and metric tensors (.pt files)
└── pyproject.toml        # Dependencies and config (uv managed)
```

**Note:** The `azimuth/` package currently contains minimal infrastructure (config constants). Most functionality lives in notebooks as this is exploratory research. Code will be extracted to package modules when patterns stabilize and reuse becomes necessary.

### Philosophy

- **Flat module organization**: We're not building enterprise software, we're putting on dad's lab coats and playing scientist. Keep modules flat in `azimuth/`. Add more as needed.
- **Notebooks for experiments**: Exploratory work happens in Jupyter notebooks. The notebook IS the lab bench.
- **Package for shared code**: Anything used more than once goes in the package. No copy-paste rot.
- **Clean but not stuffy**: Docstrings where they matter. No `# set x to 10` comments. No enterprise-Java nonsense.

## Technical Details

### Environment

- **Python:** 3.12 (pinned to match paid compute)
- **PyTorch:** 2.8 (pinned to match paid compute)
- **Development:** Local may use 3.14/2.9, but production code targets 3.12/2.8
- **Compute targets:**
  - Local: Mac with Apple Silicon (MPS), ~24GB RAM constraint
  - Cloud: NVIDIA GPUs (RTX PRO 6000, H200 SXM @ $2.29/hr, B200)

### Dtype Strategy

**For vector extraction and steering experiments:**

- **Model weights:** bfloat16 (~8GB for 4B models)
- **Forward pass activations:** bfloat16 (memory efficient, hardware-accelerated)
- **Accumulation/averaging:** float32 (numerical stability for mean calculations)
- **Final vectors:** float32 (tiny storage cost, preserves precision)

**Rationale:** We're RAM-constrained on local machines (~24GB) and want best precision we can afford. Loading models in bfloat16 saves memory and matches native dtype of modern models. Converting to float32 for averaging prevents accumulation errors when computing means across many samples. Final vectors are small enough (~450KB for 32 layers) that float32 storage is negligible.

**Model choice:** `Qwen/Qwen3-4B-Instruct-2507` — responds best to steering, excellent name

### Dependencies

- `accelerate` — Model loading and device management
- `jupyter` — Notebook environment
- `matplotlib` — Basic plotting
- `networkx` — Graph construction and analysis
- `numpy` — Numerical operations
- `pandas` — Data handling
- `plotly` — Interactive 3D visualizations (way better than matplotlib for some things)
- `python-louvain` — Community detection algorithm
- `textstat` — Flesch-Kincaid and readability metrics
- `transformers` — HuggingFace models and tokenizers
- `umap-learn` — Dimensionality reduction preserving metric structure

### Key Package Modules

**`azimuth/config.py`** (currently implemented)
- `RANDOM_SEED`: Fixed seed for reproducible dataset shuffling
- `MIN_GRADE_LEVEL_DELTA`: Minimum grade level difference for pair inclusion (4.0)
- `MIN_ARTICLE_LENGTH_REGULAR`: Minimum length for regular Wikipedia articles (1000 chars)
- `MIN_ARTICLE_LENGTH_SIMPLE`: Minimum length for Simple Wikipedia articles (500 chars)

**Planned modules** (not yet implemented, patterns still stabilizing in notebooks):
- `azimuth/models.py` — Model loading infrastructure
- `azimuth/steering.py` — Steering hooks and generation
- `azimuth/analysis.py` — Text metrics and regression analysis
- `azimuth/vectors.py` — Vector loading and manipulation
- `azimuth/visualization.py` — Plotting helpers

## Causal Metric Tensor (Current Focus)

**Reference:** Park et al. (2024) - "Linearity of Relation Decoding in Transformer Language Models"

The causal metric tensor M = Cov(γ)^-1 defines a natural geometry on semantic space based on the model's probability distribution:

**Key insight:** Token representations live in a curved, non-Euclidean space when measured by the causal metric. Standard L2 distances don't capture semantic relationships - the causal metric does.

**Geometric properties discovered:**
- Space is ~52% effective dimensionality (anisotropic, not all directions equal)
- 375,000% deviation from Euclidean geometry (strongly non-Euclidean)
- Positive curvature (κ ≈ 26.7) - sphere-like clustering, not flat or hyperbolic
- Finite diameter under causal metric with measurable typical separations
- Surprising community structure (4 equal clusters instead of hierarchical)

**Applications:**
1. Better layer selection for steering (causal norm vs L2 norm)
2. Understanding manifold boundaries (where do we fall off?)
3. Measuring steering vector quality (magnitude under causal vs Euclidean metric)
4. Geometric interpretation of model capabilities

## Research Hypotheses

### Hypothesis: Semantic Archipelago

**Prediction:** Perplexity maps will show low-PPL "islands" around stable attractors (on-manifold regions), high-PPL "oceans" between islands (off-manifold voids), and sharp transitions at boundaries.

**Test:** 2D sweeps with fine resolution (0.5 unit spacing)

### Hypothesis: Discontinuity = Manifold Boundary

**Prediction:** Grade-level discontinuities correspond to perplexity threshold crossings.

**Test:** 1D sweeps measuring both metrics simultaneously, looking for coincident jumps.

### Hypothesis: Multiple Stable Regions

**Prediction:** Extreme steering (α=20, 50, 100) might find OTHER low-PPL regions — alternative stable attractors ("semantic galaxies").

**Test:** Monitor perplexity at extreme steering coefficients. Monotonic increase = single manifold. Local minima = multiple stable regions.

## Common Patterns (from llmsonar analysis)

These patterns repeat across experimental notebooks and should live in the package:

1. **Model loading boilerplate** — Same device checks, dtype selection, eval mode
2. **Steering hooks** — Forward hook registration, vector addition, cleanup
3. **Generation wrapper** — Chat templates, special token cleanup, response extraction
4. **Text analysis** — Flesch-Kincaid, word/sentence counts, error handling
5. **Vector manipulation** — Loading, normalization, orthogonality checks
6. **Linear regression** — scipy.linregress + visualization
7. **CSV persistence** — Pandas DataFrame → CSV with consistent structure

## Vector Extraction Method

Following Chen et al. (2025), we extract steering vectors using contrastive activation averaging.

**Implementation:** See `notebooks/02_vector_extraction.ipynb`

**Extraction method:** Uses HuggingFace's `output_hidden_states=True` to capture layer activations, with attention mask to exclude padding tokens from mean pooling.

**Algorithm:**
1. For each text pair (simple, complex):
   - Tokenize and truncate to MAX_SEQ_LENGTH (default: 4096 tokens)
   - Run forward pass with `output_hidden_states=True` (no gradients, eval mode)
   - Capture hidden states at each layer (skipping embedding layer)
   - Mean-pool across sequence dimension, **excluding padding tokens via attention mask**
   - Convert to float32 and accumulate
2. After processing all pairs:
   - For each layer: compute mean of all "complex" activation vectors
   - For each layer: compute mean of all "simple" activation vectors
   - Subtract: `complexity_vector[layer] = mean(complex) - mean(simple)`
3. Analyze magnitude (L2 norm) by layer to find "best steering layer"
   - With `output_hidden_states` method: typically layer N-1 (second-to-last layer)
   - For Qwen3-4B: Layer 34 (out of 36) shows maximum magnitude
   - Interpretation: complexity/simplicity encoded near token-selection level

**Key insights:**
- Mean-pool FIRST (across tokens within each text), THEN average (across texts)
- Padding tokens must be excluded via attention mask for accurate mean pooling
- Accumulate in float32 for numerical stability, even though model is bfloat16

**Output format:** Save as `.pt` file with metadata:
```python
{
    'vectors': torch.Tensor,        # [n_layers, hidden_dim], float32
    'layer_norms': torch.Tensor,    # [n_layers], L2 norms
    'best_layer': int,              # argmax of layer_norms
    'metadata': {
        'model': str,
        'dataset': str,
        'n_pairs': int,
        'max_seq_length': int,
        'extraction_date': str,
    }
}
```

## Implemented Notebooks

### `01_datasets.ipynb`
Collects and analyzes text pairs with contrasting complexity levels.

**Datasets:**
1. **OneStopEnglish** (189 articles, 3 reading levels): Elementary vs. Advanced pairs
2. **Wikipedia** (Level-3 Vital Articles): Simple English Wikipedia vs. Regular Wikipedia

**Selection criteria:**
- Minimum grade level delta: 4.0 (configurable in `azimuth/config.py`)
- Top 20 pairs by Flesch-Kincaid delta
- Quality analysis with visualizations (histograms, scatter plots, cumulative distributions)

**Output format:**
- `{dataset}_all_pairs.csv` — Full dataset with metadata
- `{dataset}_top20_pairs.csv` — Top 20 selected pairs
- `{dataset}_top20_texts.json` — Texts only, ready for vector extraction

### `02_vector_extraction.ipynb`
Extracts steering vectors using contrastive activation averaging.

**Method:**
- Uses `output_hidden_states=True` (HuggingFace Transformers)
- Attention mask to exclude padding tokens
- bfloat16 for model/activations, float32 for accumulation
- Analyzes magnitude by layer to identify best steering layer

**Parameterization:** Config cell at top sets:
- `DATASET_PATH` — Path to `*_top20_texts.json`
- `MODEL_NAME` — HuggingFace model identifier
- `MAX_SEQ_LENGTH` — Token truncation limit (4096)
- `DEVICE` — 'auto', 'cuda', 'mps', or 'cpu'

**Output:**
- `data/vectors/complexity_{dataset}.pt` containing:
  - `vectors`: [n_layers, hidden_dim] tensor (float32)
  - `layer_norms`: L2 norms by layer
  - `best_layer`: Layer with maximum magnitude
  - `metadata`: Model name, dataset info, extraction date

### `03_causal_metric_tensor.ipynb`
Extracts the causal metric tensor M = Cov(γ)^-1 from the model's unembedding matrix (Park et al. 2024).

**Method:**
- Loads model and extracts unembedding matrix γ (vocab_size × hidden_dim)
- Computes covariance: Cov(γ) = (1/V) Σ (γᵢ - μ)(γᵢ - μ)ᵀ
- Applies Tikhonov regularization: Cov(γ) + λI (λ=1e-6 for numerical stability)
- Inverts to get metric tensor: M = (Cov(γ) + λI)^-1
- Verifies properties: symmetry, positive definiteness, condition number

**Output:**
- `data/vectors/causal_metric_tensor_qwen3_4b.pt` containing:
  - `M`: Metric tensor [hidden_dim, hidden_dim] (float32)
  - `cov_gamma`: Covariance matrix
  - `eigenvalues_cov`: For condition number analysis
  - `metadata`: Model name, regularization parameter, extraction date

**Key insight:** The causal metric defines distances in semantic space that account for the model's actual probability distributions, not just Euclidean geometry.

### `04.1_metric_properties.ipynb`
Analyzes basic geometric properties of the causal metric tensor.

**Analyses:**
1. **Effective dimensionality** via participation ratio: PR = (Σλᵢ)² / Σ(λᵢ²)
2. **Distance from Euclidean** via Frobenius norm: ||M - I||_F
3. **Token cloud extent** via sampling and greedy diameter search

**Key findings:**
- Semantic space is ~52% effective dimensionality (~1333 active dimensions)
- Strongly non-Euclidean (375,000% deviation from identity)
- Token cloud has finite diameter under causal metric

### `04.2_forman_ricci_curvature.ipynb`
Estimates discrete Ricci curvature of semantic space using Forman's combinatorial method.

**Method:**
- Samples tokens, builds k-NN graph using causal distances
- Computes Forman-Ricci curvature: κ_F(i,j) = deg(i) + deg(j) - 2 - 2·(common neighbors)
- Tests convergence across sample sizes [500, 1000, 2000, 4000, 8000]

**Key finding:**
- Positive curvature (κ ≈ 26.7) indicates sphere-like, locally clustered geometry
- Space is not flat or hyperbolic

### `04.3_community_detection.ipynb`
Identifies semantic clusters using Louvain community detection on k-NN graph.

**Method:**
- Builds k-NN graph (k=20) from causal distances
- Applies Louvain algorithm to find communities
- Analyzes modularity and size distribution

**Key finding:**
- Surprisingly symmetric structure (4 roughly equal communities)
- Not the hierarchical or power-law distribution initially expected

### `04.4_umap_visualization.ipynb`
Creates 2D and 3D visualizations of semantic space under causal metric.

**Method:**
- Samples tokens, computes pairwise causal distances
- Uses UMAP with `metric='precomputed'` to preserve causal geometry
- Validates structure with Hopkins statistic and community detection
- Creates interactive Plotly visualizations

**Features:**
- 2D and 3D embeddings
- Token ID and community-colored visualizations
- Hopkins statistic for clustering validation
- Interactive rotation/zoom in 3D plots

**Key insight:** Visualizations reveal the non-Euclidean structure - clusters and voids that wouldn't be visible in Euclidean projections.

## Common Development Commands

**Running notebooks:**
```bash
jupyter notebook notebooks/
```

**Running inline Python with uv:**
```bash
uv run python -c "import torch; print(torch.__version__)"
```

**Loading extracted vectors:**
```python
import torch
vectors = torch.load('data/vectors/complexity_wikipedia.pt')
best_layer = vectors['best_layer']  # typically 34 for Qwen3-4B
steering_vec = vectors['vectors'][best_layer]  # [hidden_dim]
```

## Notes for Future Alpha

- This project builds on insights from the llmsonar work (steering vectors as local paths through sparse manifolds, not global feature axes)
- We're specifically investigating the "catastrophic boundary" phenomenon (sharp discontinuities when leaving trained regions)
- Perplexity is the key new metric — it tells us when we've left the manifold
- **NEW DIRECTION**: Exploring causal metric tensor from Park et al. 2024 to understand the geometric structure of semantic space
- When making notebooks: keep the experimental narrative, let the story unfold, don't rush to completion
- When adding to the package: if you write it twice in notebooks, it belongs in `azimuth/`
- **Notebooks are parameterized at the top** — set dataset path, model name, batch size, etc. in config cells for reusability
- The extraction method (output_hidden_states vs. forward hooks) affects magnitude distribution but produces semantically equivalent vectors (cosine similarity >0.99)

### Tool Gotchas

**NotebookEdit with insert mode:**
- When using `NotebookEdit` with `edit_mode='insert'` and `cell_id='cell-X'`, the new cell is inserted AFTER cell-X
- If you insert multiple cells in sequence all using the same `cell_id`, they'll be inserted in REVERSE order (4321 instead of 1234)
- **Solution**: Either insert cells in reverse order, OR track the new cell IDs and use them for subsequent insertions, OR just rewrite the whole notebook with Write tool
- This bit us when adding the token cloud extent section to 04.1 - ended up with cells completely backwards!

## Final Note from Jeffery

- Sorry, forgot to mention, I'm setting us up to use `uv` for this, so it'll be `uv run python -c…` if you want to run inline Python scripts or whatever.
- Thanks for being you. Always.
