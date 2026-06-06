"""
The Thesis Forge - Advanced Multi-Stage Humanization Pipeline

A modular NLP pipeline that transforms AI-generated text to bypass
AI detection systems by addressing token probability distributions,
perplexity uniformity, n-gram predictability, and stylistic fingerprints.
"""

from humanizer.pipeline import HumanizationPipeline

__version__ = "2.0.0"
__all__ = ["HumanizationPipeline"]
