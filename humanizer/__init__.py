"""
The Thesis Forge - Advanced Multi-Stage Humanization Pipeline

A modular NLP pipeline that transforms AI-generated text to bypass
AI detection systems by addressing token probability distributions,
perplexity uniformity, n-gram predictability, and stylistic fingerprints.
"""

from humanizer.pipeline import HumanizationPipeline
from humanizer.voice_analysis import get_voice_profile
from humanizer.critic import CriticLoop
from humanizer.identity import AcademicIdentity

__version__ = "2.0.0"
__all__ = ["HumanizationPipeline", "get_voice_profile", "CriticLoop", "AcademicIdentity"]
