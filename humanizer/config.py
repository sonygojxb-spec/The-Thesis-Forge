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
        # Existing stages
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
        # New stages — Level 1 (Light): only semantic_transform enabled
        "semantic_transform_enabled": True,
        "semantic_transform_aggression": 0.1,
        "iterative_paraphrase_enabled": False,
        "iterative_paraphrase_aggression": 0.0,
        "retrieval_augmented_enabled": False,
        "retrieval_augmented_aggression": 0.0,
        "stylometric_enabled": True,
        "stylometric_aggression": 0.1,
        "perplexity_optimize_enabled": False,
        "perplexity_optimize_aggression": 0.0,
        "adversarial_enabled": False,
        "adversarial_aggression": 0.0,
        "error_injection_enabled": False,
        "error_injection_aggression": 0.0,
        "detector_optimize_enabled": False,
        "detector_optimize_aggression": 0.0,
        "classifier_enabled": False,
        # Default target perplexity profile
        "target_perplexity_mean": 60.0,
        "target_perplexity_variance": 15.0,
    },
    2: {
        # Existing stages
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
        # New stages — Level 2 (Moderate): semantic + iterative + stylometric
        "semantic_transform_enabled": True,
        "semantic_transform_aggression": 0.3,
        "iterative_paraphrase_enabled": True,
        "iterative_paraphrase_aggression": 0.2,
        "retrieval_augmented_enabled": False,
        "retrieval_augmented_aggression": 0.2,
        "stylometric_enabled": True,
        "stylometric_aggression": 0.3,
        "perplexity_optimize_enabled": False,
        "perplexity_optimize_aggression": 0.2,
        "adversarial_enabled": False,
        "adversarial_aggression": 0.2,
        "error_injection_enabled": False,
        "error_injection_aggression": 0.1,
        "detector_optimize_enabled": False,
        "detector_optimize_aggression": 0.2,
        "classifier_enabled": False,
        # Default target perplexity profile
        "target_perplexity_mean": 60.0,
        "target_perplexity_variance": 15.0,
    },
    3: {
        # Existing stages
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
        # New stages — Level 3 (Standard): add perplexity_optimize, retrieval, error_injection
        "semantic_transform_enabled": True,
        "semantic_transform_aggression": 0.5,
        "iterative_paraphrase_enabled": True,
        "iterative_paraphrase_aggression": 0.4,
        "retrieval_augmented_enabled": True,
        "retrieval_augmented_aggression": 0.4,
        "stylometric_enabled": True,
        "stylometric_aggression": 0.5,
        "perplexity_optimize_enabled": True,
        "perplexity_optimize_aggression": 0.4,
        "adversarial_enabled": False,
        "adversarial_aggression": 0.4,
        "error_injection_enabled": True,
        "error_injection_aggression": 0.3,
        "detector_optimize_enabled": False,
        "detector_optimize_aggression": 0.4,
        "classifier_enabled": False,
        # Default target perplexity profile
        "target_perplexity_mean": 60.0,
        "target_perplexity_variance": 15.0,
    },
    4: {
        # Existing stages
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
        # New stages — Level 4 (Aggressive): add adversarial, classifier, detector_optimize
        "semantic_transform_enabled": True,
        "semantic_transform_aggression": 0.7,
        "iterative_paraphrase_enabled": True,
        "iterative_paraphrase_aggression": 0.7,
        "retrieval_augmented_enabled": True,
        "retrieval_augmented_aggression": 0.6,
        "stylometric_enabled": True,
        "stylometric_aggression": 0.7,
        "perplexity_optimize_enabled": True,
        "perplexity_optimize_aggression": 0.6,
        "adversarial_enabled": True,
        "adversarial_aggression": 0.7,
        "error_injection_enabled": True,
        "error_injection_aggression": 0.5,
        "detector_optimize_enabled": True,
        "detector_optimize_aggression": 0.6,
        "classifier_enabled": True,
        # Default target perplexity profile
        "target_perplexity_mean": 60.0,
        "target_perplexity_variance": 15.0,
    },
    5: {
        # Existing stages
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
        # New stages — Level 5 (Maximum): all enabled, high aggression
        "semantic_transform_enabled": True,
        "semantic_transform_aggression": 0.9,
        "iterative_paraphrase_enabled": True,
        "iterative_paraphrase_aggression": 1.0,
        "retrieval_augmented_enabled": True,
        "retrieval_augmented_aggression": 0.9,
        "stylometric_enabled": True,
        "stylometric_aggression": 0.9,
        "perplexity_optimize_enabled": True,
        "perplexity_optimize_aggression": 0.8,
        "adversarial_enabled": True,
        "adversarial_aggression": 0.9,
        "error_injection_enabled": True,
        "error_injection_aggression": 0.7,
        "detector_optimize_enabled": True,
        "detector_optimize_aggression": 0.9,
        "classifier_enabled": True,
        # Default target perplexity profile
        "target_perplexity_mean": 60.0,
        "target_perplexity_variance": 15.0,
    },
}

# LLM Rewrite stage settings
LLM_PASS1_TEMPERATURE_BASE = 0.8
LLM_PASS2_TEMPERATURE_BASE = 0.9
LLM_TEMPERATURE_INTENSITY_FACTOR = 0.15

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
    "in conclusion", "it is evident that", "plays a crucial role",
    "it is essential to", "a myriad of", "in today's world",
    "serves as a", "it is noteworthy", "a comprehensive understanding",
    "the overarching", "holistic approach", "a nuanced understanding",
    "leveraging", "spearheading", "at the forefront",
    "groundbreaking", "cutting-edge", "paradigm shift",
    "synergy", "ecosystem", "robust framework",
    "seamless integration", "transformative", "instrumental in",
    "underpinning", "fostering", "bolstering", "catalyzing",
    "it is crucial to", "it is worth mentioning",
    "a key aspect", "plays an important role",
    "it can be argued that", "a significant impact",
    "on the other hand", "in other words",
    "as a matter of fact", "by and large",
    "at the end of the day", "in a nutshell",
    "to put it simply", "as previously mentioned",
    "it goes without saying", "as a consequence",
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
    "in conclusion": ["to sum up", "finally", "overall", ""],
    "it is evident that": ["clearly", "one can see that", "it appears", ""],
    "plays a crucial role": ["matters greatly", "is key", "is central", ""],
    "it is essential to": ["one must", "it helps to", "we need to", ""],
    "a myriad of": ["many", "a range of", "various", "several"],
    "in today's world": ["now", "currently", "these days", "at present"],
    "serves as a": ["acts as a", "functions as a", "works as a", "is a"],
    "it is noteworthy": ["notably", "worth noting", "interestingly", ""],
    "a comprehensive understanding": ["a full picture", "a clear grasp", "solid knowledge", ""],
    "the overarching": ["the main", "the broad", "the general", "the central"],
    "holistic approach": ["broad approach", "complete method", "full view", "integrated method"],
    "a nuanced understanding": ["a detailed grasp", "a careful reading", "a close look", ""],
    "leveraging": ["using", "drawing on", "employing", "making use of"],
    "spearheading": ["leading", "driving", "heading", "championing"],
    "at the forefront": ["leading", "ahead", "in front", "at the front"],
    "groundbreaking": ["pioneering", "original", "path-breaking", "novel"],
    "cutting-edge": ["advanced", "latest", "modern", "state-of-the-art"],
    "paradigm shift": ["major change", "fundamental shift", "turning point", "sea change"],
    "synergy": ["cooperation", "collaboration", "combined effect", "partnership"],
    "ecosystem": ["environment", "system", "network", "setting"],
    "robust framework": ["solid structure", "strong model", "reliable scheme", "sound basis"],
    "seamless integration": ["smooth merging", "easy combination", "natural fit", ""],
    "transformative": ["significant", "major", "far-reaching", "game-changing"],
    "instrumental in": ["key to", "important for", "central to", "vital for"],
    "underpinning": ["supporting", "underlying", "behind", "at the base of"],
    "fostering": ["encouraging", "promoting", "cultivating", "nurturing"],
    "bolstering": ["strengthening", "supporting", "reinforcing", "boosting"],
    "catalyzing": ["triggering", "sparking", "driving", "prompting"],
    "it is crucial to": ["one must", "it matters to", "we should", ""],
    "it is worth mentioning": ["notably", "it helps to note", "one point is", ""],
    "a key aspect": ["one part", "an important point", "a main element", ""],
    "plays an important role": ["matters", "is significant", "contributes", ""],
    "it can be argued that": ["one might say", "arguably", "perhaps", ""],
    "a significant impact": ["a real effect", "a strong influence", "a clear bearing", ""],
    "on the other hand": ["but", "yet", "however", "alternatively"],
    "in other words": ["that is", "put differently", "meaning", ""],
    "as a matter of fact": ["in fact", "actually", "indeed", ""],
    "by and large": ["mostly", "generally", "on the whole", ""],
    "at the end of the day": ["ultimately", "in the end", "finally", ""],
    "in a nutshell": ["briefly", "in short", "simply put", ""],
    "to put it simply": ["simply", "in short", "basically", ""],
    "as previously mentioned": ["as noted", "as stated", "as discussed", ""],
    "it goes without saying": ["clearly", "obviously", "naturally", ""],
    "as a consequence": ["so", "therefore", "as a result", "thus"],
}

# Critic Loop settings
CRITIC_DEFAULT_THRESHOLD = 40
CRITIC_MAX_RETRIES = 3

# Voice Analysis marker lists
VOICE_HEDGES = [
    "might", "perhaps", "seems to", "could", "arguably",
    "it appears", "one might", "possibly", "may", "likely",
    "it seems", "suggest", "to some extent",
]

VOICE_BOOSTERS = [
    "clearly", "definitely", "certainly", "undoubtedly",
    "obviously", "precisely", "without doubt", "evidently",
    "unquestionably", "absolutely",
]

VOICE_ENGAGEMENT_MARKERS = [
    "one might argue",
    "alternatively", "some scholars",
    "it could be contended", "a counter-argument",
    "from another perspective",
]

# Academic Identity defaults
ACADEMIC_ROLES = [
    "PhD student", "Postdoctoral researcher", "Assistant Professor",
    "Associate Professor", "Professor", "Research Fellow",
    "Lecturer", "Graduate student", "Research Scholar",
]

ACADEMIC_FIELDS = [
    "Computer Science", "Physics", "Biology", "Chemistry",
    "Mathematics", "Psychology", "Economics", "Literature",
    "Engineering", "Medicine", "Sociology", "Philosophy",
    "Political Science", "Environmental Science", "Linguistics",
]

STYLE_PREFERENCES = ["formal", "semi-formal", "conversational"]

# Indian English spelling preferences (American -> British)
INDIAN_ENGLISH_PREFERENCES = {
    "utilize": "utilise",
    "utilize": "utilise",
    "organize": "organise",
    "recognize": "recognise",
    "analyze": "analyse",
    "summarize": "summarise",
    "maximize": "maximise",
    "minimize": "minimise",
    "optimize": "optimise",
    "emphasize": "emphasise",
    "standardize": "standardise",
    "prioritize": "prioritise",
    "customize": "customise",
    "categorize": "categorise",
    "characterize": "characterise",
    "hypothesize": "hypothesise",
    "color": "colour",
    "behavior": "behaviour",
    "favor": "favour",
    "honor": "honour",
    "labor": "labour",
    "neighbor": "neighbour",
    "center": "centre",
    "fiber": "fibre",
    "meter": "metre",
    "defense": "defence",
    "license": "licence",
    "offense": "offence",
    "program": "programme",
    "catalog": "catalogue",
    "dialog": "dialogue",
}

# Indian English discourse markers typical in academic writing
INDIAN_DISCOURSE_MARKERS = [
    "as such",
    "in this regard",
    "to that end",
    "one may note that",
    "it is pertinent to mention",
    "needless to say",
    "in view of the above",
    "keeping this in mind",
    "on the whole",
    "for the most part",
    "to a large extent",
    "in the Indian context",
    "broadly speaking",
]
