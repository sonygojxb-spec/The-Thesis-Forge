# The Thesis Forge - Advanced AI Humanizer

A multi-stage humanization pipeline that transforms AI-generated academic text to bypass AI detection systems like Turnitin by addressing the core patterns these detectors look for:

- **Token probability distributions** - AI text uses uniformly high-probability tokens
- **Sentence-level perplexity uniformity** - AI maintains consistent complexity; humans vary wildly
- **N-gram predictability** - AI follows predictable word sequences
- **Stylistic fingerprints** - AI has uniform sentence rhythm and limited vocabulary diversity

## Architecture

The application is built as a modular Python package (`humanizer/`) with a Streamlit frontend.

### Pipeline Stages

| Stage | Module | Type | Description |
|-------|--------|------|-------------|
| 1 | `stage_structural.py` | Deterministic/NLP | Sentence splitting, merging, clause reordering, paragraph restructuring using spaCy |
| 2 | `stage_lexical.py` | NLP | Controlled vocabulary injection using WordNet and word frequency data |
| 3 | `stage_llm_rewrite.py` | LLM (streaming) | Multi-pass rewriting with different temperatures and prompts |
| 4 | `stage_perplexity.py` | NLP | Deliberately varies sentence complexity to mimic human patterns |
| 5 | `stage_postprocess.py` | NLP | Removes AI fingerprints, eliminates AI transition words, injects natural imperfections |

### Intensity Levels

- **Level 1**: Structural + light lexical changes only. Minimal transformation.
- **Level 2**: Adds single-pass LLM rewrite with moderate settings.
- **Level 3**: All stages active at moderate aggression.
- **Level 4**: All stages at high aggression (recommended for Turnitin).
- **Level 5**: Maximum transformation across all stages.

### Module Overview

```
humanizer/
  __init__.py          - Package initialization
  config.py            - Centralized configuration (API keys, models, settings)
  text_analysis.py     - Text analytics (readability, AI risk scoring)
  stage_structural.py  - Stage 1: Structural variation
  stage_lexical.py     - Stage 2: Vocabulary injection
  stage_llm_rewrite.py - Stage 3: Multi-pass LLM rewriting
  stage_perplexity.py  - Stage 4: Perplexity variance injection
  stage_postprocess.py - Stage 5: Final cleanup
  pipeline.py          - Pipeline orchestrator
app.py                 - Streamlit frontend
```

## Installation

### Prerequisites

- Python 3.9+
- pip

### Setup

1. Install Python dependencies:

```bash
pip install -r requirements.txt
```

2. Download the spaCy language model:

```bash
python -m spacy download en_core_web_sm
```

3. Download NLTK data (WordNet):

```bash
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

### API Configuration

The app uses an OpenAI-compatible API endpoint. Edit `humanizer/config.py` to set your API key:

```python
API_KEY = "your-api-key-here"
BASE_URL = "https://api.freemodel.dev"
```

## Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

### Features

- **Model Selection**: Choose from gpt-4o, gpt-4o-mini, openai-t1-sg, openai-t2-sg
- **Intensity Slider**: Control how aggressively text is transformed (1-5)
- **Stage Toggles**: Enable/disable individual pipeline stages
- **Real-time Progress**: See which stage is currently running
- **Before/After Analytics**: Compare sentence count, average length, grade level, vocabulary diversity, and estimated AI detection risk
- **Streaming Output**: Watch the LLM rewrite in real-time
- **Download Button**: Export the humanized text

## How It Works

### Why AI Detection Catches Standard Rewrites

Simple paraphrasing tools fail because they preserve:
1. Uniform sentence length patterns
2. High-probability token sequences
3. Consistent perplexity across sentences
4. AI-favorite transition words (Moreover, Furthermore, etc.)

### How This Pipeline Defeats Detection

1. **Structural Variation** breaks the uniform rhythm by splitting/merging sentences and reordering clauses
2. **Lexical Injection** replaces predictable high-frequency words with less common alternatives
3. **LLM Rewriting** (two passes) rewrites for natural flow and adds authentic academic voice
4. **Perplexity Variance** deliberately makes some sentences simple and others complex
5. **Post-processing** removes any remaining AI fingerprints and adds natural imperfections

The combination of all five stages addresses every major signal that AI detectors use, producing text that matches the statistical patterns of genuine human writing.
