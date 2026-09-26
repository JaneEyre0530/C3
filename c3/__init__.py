"""Public C3 algorithm package."""

from .core import C3Result, apply_c3, policy_confidence
from .events import detect_events, normalize_query, trigram_jaccard

__all__ = [
    "C3Result",
    "apply_c3",
    "detect_events",
    "normalize_query",
    "policy_confidence",
    "trigram_jaccard",
]
