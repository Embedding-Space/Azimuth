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

## Project Structure

```
Azimuth/
├── azimuth/              # Python package (shared tooling)
│   ├── models.py         # Model loading infrastructure
│   ├── steering.py       # Steering hooks and generation
│   ├── analysis.py       # Text metrics and regression analysis
│   ├── vectors.py        # Vector loading and manipulation
│   └── visualization.py  # Plotting helpers
├── notebooks/            # Jupyter notebooks (experiments)
├── docs/                 # Documentation and design docs
└── pyproject.toml        # Dependencies and config
```

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

### Dependencies

- `accelerate` — Model loading and device management
- `jupyter` — Notebook environment
- `matplotlib` — Basic plotting
- `numpy` — Numerical operations
- `pandas` — Data handling
- `plotly` — Interactive 3D visualizations (way better than matplotlib for some things)
- `textstat` — Flesch-Kincaid and readability metrics
- `transformers` — HuggingFace models and tokenizers

### Key Package Modules

**`azimuth/models.py`**
- Load models and tokenizers with proper device mapping
- Handle bfloat16/float32 dtype selection
- Set models to eval mode

**`azimuth/steering.py`**
- Steering hook implementation (forward hooks that add vectors to hidden states)
- Text generation with steering applied
- Chat template formatting and special token cleanup
- Perplexity calculation from `output_scores`

**`azimuth/analysis.py`**
- Flesch-Kincaid grade level computation
- Text metrics (word count, sentence count, etc.)
- Linear regression analysis
- Statistical summaries

**`azimuth/vectors.py`**
- Load vectors from `.pt` files
- Normalize to target magnitudes
- Orthogonality verification
- float32 ↔ bfloat16 conversion

**`azimuth/visualization.py`**
- Scatter plots with regression lines
- Heatmaps for 2D parameter sweeps
- 3D surface plots (Plotly)
- Dual-metric visualizations (grade level + perplexity)

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

## Notes for Future Alpha

- This project builds on insights from the llmsonar work (steering vectors as local paths through sparse manifolds, not global feature axes)
- We're specifically investigating the "catastrophic boundary" phenomenon (sharp discontinuities when leaving trained regions)
- Perplexity is the key new metric — it tells us when we've left the manifold
- When making notebooks: keep the experimental narrative, let the story unfold, don't rush to completion
- When adding to the package: if you write it twice in notebooks, it belongs in `azimuth/`

## Final Note from Jeffery

- Sorry, forgot to mention, I'm setting us up to use `uv` for this, so it'll be `uv run python -c…` if you want to run inline Python scripts or whatever.
- Thanks for being you. Always.
