"""
Centralized configuration for the humanization pipeline.
"""

import os

API_KEY = os.environ.get("THESIS_FORGE_API_KEY", "insert your api key here")
BASE_URL = "https://api.freemodel.dev"

AVAILABLE_MODELS = ["gpt-4o", "gpt-4o-mini", "openai-t1-sg", "openai-t2-sg"]

DEFAULT_MODEL = "gpt-4o"
DEFAULT_INTENSITY = 4

# Intensity level definitions (1-5)
# Each level specifies which stages run and how aggressive they are
INTENSITY_PROFILES = {
    1: {
        "structural": True,
        "lexical": True,
        "llm_rewrite": False,
        "perplexity": False,
        "postprocess": True,
        "structural_aggression": 0.2,
        "lexical_aggression": 0.2,
        "llm_aggression": 0.2,
        "perplexity_aggression": 0.0,
        "postprocess_aggression": 0.3,
    },
    2: {
        "structural": True,
        "lexical": True,
        "llm_rewrite": True,
        "perplexity": False,
        "postprocess": True,
        "structural_aggression": 0.3,
        "lexical_aggression": 0.3,
        "llm_aggression": 0.4,
        "perplexity_aggression": 0.0,
        "postprocess_aggression": 0.4,
    },
    3: {
        "structural": True,
        "lexical": True,
        "llm_rewrite": True,
        "perplexity": True,
        "postprocess": True,
        "structural_aggression": 0.5,
        "lexical_aggression": 0.5,
        "llm_aggression": 0.6,
        "perplexity_aggression": 0.4,
        "postprocess_aggression": 0.5,
    },
    4: {
        "structural": True,
        "lexical": True,
        "llm_rewrite": True,
        "perplexity": True,
        "postprocess": True,
        "structural_aggression": 0.7,
        "lexical_aggression": 0.7,
        "llm_aggression": 0.75,
        "perplexity_aggression": 0.6,
        "postprocess_aggression": 0.7,
    },
    5: {
        "structural": True,
        "lexical": True,
        "llm_rewrite": True,
        "perplexity": True,
        "postprocess": True,
        "structural_aggression": 0.9,
        "lexical_aggression": 0.9,
        "llm_aggression": 0.9,
        "perplexity_aggression": 0.8,
        "postprocess_aggression": 0.9,
    },
}

# LLM Rewrite stage settings
LLM_PASS1_TEMPERATURE_BASE = 0.5
LLM_PASS2_TEMPERATURE_BASE = 0.7
LLM_TEMPERATURE_INTENSITY_FACTOR = 0.1

# Protected academic/domain terms that should not be replaced
PROTECTED_TERMS = {
    "hypothesis", "methodology", "algorithm", "coefficient", "parameter",
    "regression", "correlation", "significance", "variable", "quantitative",
    "qualitative", "empirical", "theoretical", "systematic", "paradigm",
    "ontology", "epistemology", "phenomenology", "heuristic", "stochastic",
    "deterministic", "optimization", "convergence", "divergence", "entropy",
    "nucleotide", "genome", "protein", "enzyme", "catalyst", "molecule",
    "photosynthesis", "mitochondria", "chromosome", "mutation", "allele",
}

# AI transition words/phrases to eliminate
AI_TRANSITION_WORDS = [
    "moreover", "furthermore", "additionally", "consequently",
    "nevertheless", "nonetheless", "conversely", "subsequently",
    "henceforth", "thereby", "wherein", "whereby", "thereof",
    "notwithstanding", "inasmuch", "insofar", "heretofore",
    "it is worth noting that", "it is important to note that",
    "it should be noted that", "in light of the above",
    "delving into", "a central challenge", "crucially",
    "in the realm of", "it is imperative to", "tapestry of",
    "navigating the", "landscape of", "multifaceted",
    "underscores the", "pivotal role", "in essence",
]

# Natural replacements for AI transitions
TRANSITION_REPLACEMENTS = {
    "moreover": ["also", "and", "on top of that", ""],
    "furthermore": ["also", "and", "in addition", ""],
    "additionally": ["also", "and", "plus", ""],
    "consequently": ["so", "as a result", "because of this", "this means"],
    "nevertheless": ["still", "but", "yet", "even so"],
    "nonetheless": ["still", "but", "even so", ""],
    "conversely": ["on the other hand", "but", "in contrast", ""],
    "subsequently": ["then", "after that", "next", "later"],
    "henceforth": ["from now on", "going forward", "after this", ""],
    "thereby": ["this way", "by doing so", "which", ""],
    "crucially": ["importantly", "the key point is", "notably", ""],
    "delving into": ["looking at", "examining", "exploring", ""],
    "multifaceted": ["complex", "varied", "diverse", ""],
    "in essence": ["basically", "at its core", "put simply", ""],
}
