"""Task-light bad-pattern event construction for C3.

The detector consumes parsed turn records from a caller. It deliberately does
not parse a model-specific prompt or call an environment service. A turn record
contains ``action`` (``search``/``answer``/other), ``query`` for search turns,
and optional ``parse_ok``. Returned events use the same half-open token spans
as :func:`c3.core.apply_c3`.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any


_ZERO_WIDTH = {"\u200b", "\u200c", "\u200d", "\ufeff"}


def _flag(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "none", "nan"}
    return bool(value)


def normalize_query(value: Any) -> str:
    """Normalize a search query for exact and character-trigram comparison."""

    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = "".join(char for char in text if char not in _ZERO_WIDTH)
    text = "".join(" " if unicodedata.category(char).startswith("P") else char for char in text)
    return re.sub(r"\s+", " ", text).strip()


def _trigrams(value: str) -> set[str]:
    padded = f"  {value}  "
    return {padded[index : index + 3] for index in range(max(0, len(padded) - 2))}


def trigram_jaccard(left: str, right: str) -> float:
    """Return character-trigram Jaccard similarity in ``[0, 1]``."""

    left_set, right_set = _trigrams(left), _trigrams(right)
    if not left_set and not right_set:
        return 1.0
    return len(left_set & right_set) / max(len(left_set | right_set), 1)


def _span(start: int, end: int, action_span: Any = None) -> dict[str, Any] | None:
    try:
        first, last = int(start), int(end)
    except (TypeError, ValueError):
        return None
    if last <= first:
        return None
    return {"start": first, "end": last}


def _confidence_target(action_span: Any) -> dict[str, int] | None:
    if not isinstance(action_span, Mapping):
        return None
    try:
        start, end = int(action_span["start"]), int(action_span["end"])
    except (KeyError, TypeError, ValueError):
        return None
    return {"start": start, "end": end} if end > start else None


def _event(pattern: str, value: float, target: dict[str, Any], action_span: Any) -> dict[str, Any]:
    event: dict[str, Any] = {"pattern": pattern, "value": value, "target": target}
    confidence_target = _confidence_target(action_span)
    if confidence_target is not None:
        event["confidence_target"] = confidence_target
    return event


def detect_events(
    turns: Sequence[Mapping[str, Any]],
    turn_boundaries: Sequence[int],
    *,
    action_spans: Sequence[Mapping[str, int] | None] | None = None,
    near_threshold: float = 0.9,
    near_min_chars: int = 8,
) -> list[dict[str, Any]]:
    """Detect the three public bad patterns from parsed turns.

    Strict-format-only failures are intentionally ignored when an executable
    action was recovered. Exact query repeats take precedence over near-loop
    events. Every event receives the full responsible-turn span as ``target``;
    an available action span is copied to ``confidence_target``.
    """

    if len(turns) != len(turn_boundaries):
        raise ValueError("turns and turn_boundaries must have equal lengths")
    if not 0.0 <= float(near_threshold) <= 1.0:
        raise ValueError("near_threshold must be in [0, 1]")
    if near_min_chars < 0:
        raise ValueError("near_min_chars must be non-negative")
    if action_spans is not None and len(action_spans) != len(turns):
        raise ValueError("action_spans and turns must have equal lengths")

    events: list[dict[str, Any]] = []
    previous_queries: list[str] = []
    has_search = False
    has_answer = False
    for index, (turn, boundary) in enumerate(zip(turns, turn_boundaries, strict=True)):
        previous_boundary = int(turn_boundaries[index - 1]) if index else -1
        action_span = action_spans[index] if action_spans else None
        target = _span(previous_boundary + 1, int(boundary) + 1)
        if target is None:
            continue

        action = str(turn.get("action") or "invalid").strip().lower()
        parse_ok = _flag(turn.get("parse_ok"), default=True)
        executable = parse_ok and action in {"search", "answer"}
        if not executable or has_answer:
            pattern = "invalid_action_after_answer" if has_answer and executable else "invalid_action"
            events.append(_event(pattern, -1.0, target, action_span))
            continue

        if action == "answer":
            has_answer = True
            continue

        query = normalize_query(turn.get("query", turn.get("action_text", "")))
        if query:
            exact = query in previous_queries
            if exact:
                events.append(_event("query_loop_exact", -1.0, target, action_span))
            elif len(query) >= near_min_chars and any(
                len(previous) >= near_min_chars and trigram_jaccard(query, previous) >= near_threshold
                for previous in previous_queries
            ):
                events.append(_event("query_loop_near", -0.5, target, action_span))
            previous_queries.append(query)
        has_search = True
    return events
