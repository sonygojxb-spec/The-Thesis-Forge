# The Thesis Forge - Ultimate AI Humanizer

[![Tests](https://img.shields.io/badge/tests-344%20passing-brightgreen)](tests/) [![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://python.org) [![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A 13-stage humanization pipeline that transforms AI-generated academic text to statistically match genuine human writing while preserving meaning and academic accuracy. Built with a measurement-and-fallback architecture that ensures every transformation is safe — bad candidates are discarded, never returned.

## What Makes This "Ultimate"

Standard paraphrasers fail because they only change surface words. AI detectors look at deeper statistical patterns: token probability distributions, perplexity uniformity, stylometric fingerprints, and n-gram predictability.

The Ultimate Humanizer attacks **all** of these signals simultaneously through 9 advanced capabilities layered on top of the original 5-stage pipeline:

| # | Capability | Type | What It Does |
|---|-----------|------|--------------|
| 1 | **Iterative Paraphrasing** | LLM | Multi-pass progressive divergence from AI phrasing (1-5 passes based on intensity) |
| 2 | **Stylometric Obfuscation** | NLP | Disrupts sentence rhythm, function-word distribution, punctuation patterns, vocabulary richness |
| 3 | **Perplexity Optimization** | NLP | Tunes text toward a target human-like perplexity/burstiness profile |
| 4 | **Adversarial Rewriting** | LLM | Rewrites specifically to reduce detector signals, verified by scoring |
| 5 | **Human-Like Error Injection** | NLP | Controlled typos, punctuation variations, informal constructions (≤5% of words) |
| 6 | **Semantic-Preserving Transforms** | NLP/LLM | Surface-form changes with a strict 0.90 similarity gate |
| 7 | **Retrieval-Augmented Humanization** | LLM | Grounds rewrites in real human-written reference passages |
| 8 | **Detector-Aware Optimization** | Closed-loop | Iteratively minimizes detection risk against a classifier (bounded 1-20 iterations) |
| 9 | **Transformer-Based Classifier** | ML Model | RoBERTa-style AI-text detector providing risk scores for optimization |

## Architecture

### 13-Stage Pipeline (Canonical Order)

```
Input Text
  │
  ├─ 1.  Structural Variation        (NLP - spaCy)
  ├─ 2.  Lexical Injection           (NLP - WordNet)
  ├─ 3.  Semantic Transformer        (NLP/LLM - 0.90 similarity gate)
  ├─ 4.  Iterative Paraphraser       (LLM - 1-5 passes)
  ├─ 5.  LLM Rewrite                 (LLM - multi-pass streaming)
  ├─ 6.  Retrieval-Augmented         (LLM + vector retrieval)
  ├─ 7.  Stylometric Obfuscator      (NLP - distributional shifts)
  ├─ 8.  Perplexity Variance         (NLP - complexity variation)
  ├─ 9.  Perplexity Optimizer        (NLP - target profile tuning)
  ├─ 10. Adversarial Rewriter        (LLM - detector evasion)
  ├─ 11. Error Injector              (NLP - controlled imperfections)
  ├─ 12. Post-processing             (NLP - AI fingerprint removal)
  └─ 13. Detector Optimizer          (Closed-loop - minimize detection risk)
  │
  ├─ Final Meaning-Preservation Check (similarity + protected spans)
  │
  Output Text + Analytics
```

### Safety Architecture: Measurement & Fallback

Every stage is wrapped in a **discard-on-violation** guard:
- **Semantic Similarity** is measured (embedding-based) after each transformation
- If similarity drops below the stage's floor (0.80-0.90), the candidate is **discarded** and the input is returned unchanged
- **Protected Terms** (academic vocabulary, numbers, citations) are preserved with identical occurrence counts
- **Detection Risk** is scored before and after — a rewrite that increases risk is rejected

This means you can run at maximum intensity without fear of destroying meaning.

### Key Components

| Component | Purpose |
|-----------|---------|
| `SimilarityEvaluator` | Embedding-based semantic comparison (all-MiniLM-L6-v2) with lexical-proxy fallback |
| `Classifier` | RoBERTa-style AI-text detector (0-100 risk score) with heuristic fallback |
| `ProtectedSpanGuard` | Preserves academic terms, numbers, citations, and quoted content |
| `RetrievalService` | Vector-indexed human-written reference corpus for style grounding |
| `ConfigSerializer` | JSON configuration persistence with round-trip validation |
| `DetectorOptimizer` | Closed-loop optimization using the Classifier as discriminator |

## Installation

### Prerequisites

- Python 3.9+
- pip

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/hariomaggarwal11/The-Thesis-Forge.git
cd The-Thesis-Forge

# Install dependencies
pip install -r requirements.txt

# Download NLP models
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

### Optional: Full ML capabilities

For the transformer classifier and embedding-based similarity (recommended):

```bash
pip install sentence-transformers transformers torch
```

> **Note:** The system degrades gracefully without these. The original 5-stage pipeline works perfectly with just the base requirements, using heuristic scoring and lexical similarity as fallbacks.

### API Configuration

The app uses an OpenAI-compatible API endpoint for LLM-backed stages.

**Option 1: Environment variable (recommended)**
```bash
export THESIS_FORGE_API_KEY="your-api-key-here"
```

**Option 2: Edit config directly**
```python
# humanizer/config.py
API_KEY = "your-api-key-here"
BASE_URL = "https://api.freemodel.dev"
```

## Usage

### Run the App

```bash
streamlit run app.py
```

### Frontend Features

- **Model Selection** — Choose from gpt-4o, gpt-4o-mini, and other compatible models
- **Intensity Slider (1-5)** — Single control governing all 13 stages
- **Per-Stage Toggles** — Enable/disable any of the 13 stages individually
- **Target Perplexity Profile** — Configure target mean and variance for perplexity optimization
- **Real-time Progress** — See each stage's running/complete/error status live
- **Before/After Analytics** — Detection risk score (0-100) and semantic similarity (0.0-1.0)
- **Config Save/Load** — Export and import your configuration as JSON
- **Export** — Download the humanized text as a file
- **Disclaimer** — Detection risk scores are estimates, not guarantees

### Intensity Levels

| Level | Behavior |
|-------|----------|
| **1** | Light touch — structural + lexical + semantic transforms only |
| **2** | Moderate — adds iterative paraphrasing and stylometric obfuscation |
| **3** | Balanced — all stages active at moderate aggression |
| **4** | Aggressive — high aggression + adversarial rewriting + detector optimization |
| **5** | Maximum — all stages at full intensity with closed-loop optimization |

### Programmatic Usage

```python
from humanizer.pipeline import HumanizationPipeline
from humanizer.config import INTENSITY_PROFILES

pipeline = HumanizationPipeline(
    intensity=4,
    model="gpt-4o",
    seed=42,
    progress_callback=lambda stage, status: print(f"{stage}: {status}")
)

result = pipeline.process(ai_generated_text)
print(result)
```

## Module Structure

```
humanizer/
├── __init__.py                    # Package initialization
├── config.py                      # Configuration, intensity profiles, protected terms
├── config_serializer.py           # JSON config persistence with validation
├── pipeline.py                    # 13-stage pipeline orchestrator
├── results.py                     # StageResult + TargetPerplexityProfile dataclasses
├── similarity.py                  # SimilarityEvaluator (embeddings + lexical fallback)
├── classifier.py                  # Transformer Classifier + detection_risk_score helper
├── protected_spans.py             # ProtectedSpanGuard (terms, numbers, citations, quotes)
├── retrieval.py                   # RetrievalService + ReferenceEntry + vector index
├── text_analysis.py               # Readability, perplexity, AI risk heuristics
├── stage_structural.py            # Stage 1: Structural variation (spaCy)
├── stage_lexical.py               # Stage 2: Vocabulary injection (WordNet)
├── stage_semantic.py              # Stage 3: Semantic-preserving transforms (0.90 gate)
├── stage_iterative.py             # Stage 4: Iterative paraphrasing (1-5 LLM passes)
├── stage_llm_rewrite.py           # Stage 5: Multi-pass LLM rewriting
├── stage_retrieval_augmented.py   # Stage 6: Retrieval-augmented humanization
├── stage_stylometric.py           # Stage 7: Stylometric obfuscation
├── stage_perplexity.py            # Stage 8: Perplexity variance injection
├── stage_perplexity_optimize.py   # Stage 9: Perplexity profile optimization
├── stage_adversarial.py           # Stage 10: Adversarial rewriting
├── stage_error_injector.py        # Stage 11: Human-like error injection
├── stage_postprocess.py           # Stage 12: Post-processing cleanup
└── stage_detector_optimizer.py    # Stage 13: Closed-loop detector optimization

tests/
├── conftest.py                    # Shared fixtures (fakes for LLM, similarity, classifier)
├── strategies.py                  # Hypothesis strategies (academic text, aggression, configs)
├── test_protected_spans.py        # ProtectedSpanGuard property + unit tests
├── test_similarity.py             # SimilarityEvaluator property + integration tests
├── test_classifier.py             # Classifier property + example tests
├── test_semantic.py               # SemanticTransformer tests
├── test_iterative.py              # IterativeParaphraser tests
├── test_retrieval.py              # RetrievalService + RetrievalAugmented tests
├── test_stylometric.py            # StylometricObfuscator tests
├── test_perplexity_optimize.py    # PerplexityOptimizer tests
├── test_adversarial.py            # AdversarialRewriter tests
├── test_error_injector.py         # ErrorInjector tests
├── test_detector_optimizer.py     # DetectorOptimizer tests
├── test_config_serializer.py      # ConfigSerializer round-trip + validation tests
├── test_pipeline.py               # Pipeline integration property tests
├── test_pipeline_examples.py      # Pipeline example + progress + warning tests
└── test_e2e.py                    # End-to-end integration test

app.py                             # Streamlit frontend
requirements.txt                   # Runtime dependencies
requirements-dev.txt               # Dev/test dependencies (hypothesis, pytest, faiss-cpu)
```

## Testing

### Run all tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -v --timeout=120
```

### Run a specific property test

```bash
python -m pytest tests/test_pipeline.py -v -k "test_pipeline_deterministic_order"
```

### Test coverage

- **344 tests** covering all 23 correctness properties from the design document
- Property-based tests use `hypothesis` with minimum 100 examples each
- All LLM/model dependencies are mocked for deterministic offline testing
- Integration tests exercise real models when available (marked, skippable)

### Correctness Properties Tested

| # | Property | What It Guarantees |
|---|----------|-------------------|
| 1 | Protected-span invariance | Academic terms, numbers, citations never altered |
| 2 | Numeric/citation preservation | Identical values and occurrence counts |
| 3 | Similarity-floor guarantee | No stage returns text below its meaning threshold |
| 4 | Seed/model determinism | Same input + seed = same output |
| 5 | Score-range validity | Risk scores 0-100, similarity 0.0-1.0 |
| 6 | Paraphrase divergence | Output always differs from input |
| 7 | Pass-count monotonicity | Higher aggression = more passes |
| 8-9 | Stylometric shift/monotonicity | Measurable attribute changes scale with aggression |
| 10 | Perplexity distance non-increase | Output is never further from target than input |
| 11-12 | Adversarial risk/change monotonicity | Risk never increases; changes scale with aggression |
| 13 | Error-injection bound | ≤5% of words altered, monotonic in aggression |
| 14-15 | Retrieval ranking + verbatim bound | Top-k ordering correct; no >8-word verbatim copies |
| 16 | Optimizer selection/iteration bound | Lowest-risk valid candidate within bounded iterations |
| 17 | Classifier invalid-input rejection | Empty/over-length inputs rejected cleanly |
| 18 | Pipeline deterministic order | Same config = same stage execution order |
| 19-21 | Intensity structure/monotonicity/overrides | Profiles well-formed; toggles work correctly |
| 22-23 | Config round-trip + invalid-field rejection | Serialize/deserialize is lossless; bad input rejected |

## Graceful Degradation

The system is designed to work at multiple capability levels:

| Dependencies Available | What Works |
|-----------------------|-----------|
| Base only (spaCy, nltk, requests) | Original 5-stage pipeline with heuristic risk scoring |
| + `sentence-transformers` | Embedding-based similarity (all 13 stages with meaning gates) |
| + `transformers` + `torch` | Transformer classifier + detector-aware optimization |
| + `faiss-cpu` (optional) | Faster retrieval for large reference corpora |

When optional dependencies are absent, the system automatically falls back to heuristic alternatives and surfaces a notice — it never crashes.

## Configuration Persistence

Save and load your pipeline configuration as JSON:

```python
from humanizer.config_serializer import ConfigSerializer, PipelineConfig

# Save
config = PipelineConfig(intensity=4, stage_toggles={...}, ...)
json_str = ConfigSerializer.serialize(config)

# Load
config = ConfigSerializer.deserialize(json_str)
# Raises ConfigError("field_name") on invalid input
```

The config format supports: intensity level (1-5), all stage toggles, target perplexity profile, and per-stage aggression values. Round-trip equivalence is guaranteed.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Ensure all tests pass: `python -m pytest tests/ -v --timeout=120`
4. Submit a pull request

## License

MIT
