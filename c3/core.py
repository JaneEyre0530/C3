"""C3 actor-side advantage modulation.

The implementation is deliberately independent of a trainer, rollout engine,
environment, or experiment platform.  A caller supplies the already-computed
environment advantages and verifier events for one response. C3 changes only
the actor advantages; critic returns and raw rewards stay untouched.

Event spans use half-open token offsets: ``target.start`` is included and
``target.end`` is excluded.  This is the same convention used by common token
alignment APIs and prevents an off-by-one token from being silently trained.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Mapping, Sequence

import torch


@dataclass
class C3Result:
    """Result of applying C3 to a single response."""

    advantages: torch.Tensor
    metrics: dict[str, float] = field(default_factory=dict)
    applied_events: list[dict[str, Any]] = field(default_factory=list)


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.numel() == 0:
            return None
        return float(value.detach().float().mean().cpu().item())
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if isfinite(result) else None


def _event_slice(event: Mapping[str, Any], length: int) -> slice | None:
    target = event.get("target")
    target = target if isinstance(target, Mapping) else event
    start = _as_float(target.get("start"))
    end = _as_float(target.get("end"))
    if start is not None and end is not None and end > start:
        first = max(0, min(int(start), length))
        last = max(0, min(int(end), length))
        return slice(first, last) if last > first else None
    index = _as_float(target.get("index"))
    if index is None:
        return None
    token = int(index)
    return slice(token, token + 1) if 0 <= token < length else None


def _validate_vector(name: str, value: torch.Tensor, length: int | None = None) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a torch.Tensor")
    result = value.reshape(-1)
    if length is not None and result.numel() != length:
        raise ValueError(f"{name} has {result.numel()} tokens; expected {length}")
    return result


def policy_confidence(
    policy_log_probs: torch.Tensor,
    loss_mask: torch.Tensor,
    events: Sequence[Mapping[str, Any]],
    *,
    mode: str = "span_prob",
) -> torch.Tensor:
    """Build detached per-token confidence for C3.

    ``token_prob`` keeps each token's probability.  ``span_prob`` uses the
    geometric mean (inverse perplexity) over the offending action span, while
    ``span_mean_prob`` uses the arithmetic mean of token probabilities.  A
    ``confidence_target`` on an event can provide a narrower action span than
    the event's full verifier span.
    """

    log_probs = _validate_vector("policy_log_probs", policy_log_probs).detach().float()
    active = _validate_vector("loss_mask", loss_mask, log_probs.numel()).detach().bool()
    log_probs = log_probs.clamp(min=-20.0, max=0.0)
    if mode == "token_prob":
        return log_probs.exp()
    if mode not in {"span_prob", "span_mean_prob"}:
        raise ValueError(f"unknown confidence mode: {mode!r}")

    confidence = torch.zeros_like(log_probs)
    for event in events:
        event_span = _event_slice(event, log_probs.numel())
        if event_span is None:
            continue
        confidence_event = event
        target = event.get("confidence_target")
        if isinstance(target, Mapping):
            confidence_event = dict(event)
            confidence_event["target"] = target
        confidence_span = _event_slice(confidence_event, log_probs.numel()) or event_span
        values = log_probs[confidence_span][active[confidence_span]]
        if values.numel() == 0:
            continue
        span_confidence = values.exp().mean() if mode == "span_mean_prob" else values.mean().exp()
        confidence[event_span] = torch.maximum(
            confidence[event_span], span_confidence.expand_as(confidence[event_span])
        )
    return confidence


def apply_c3(
    advantage: torch.Tensor,
    loss_mask: torch.Tensor,
    events: Sequence[Mapping[str, Any]],
    *,
    reversal_scale: float = 0.0,
    confidence_reversal_scale: float = 1.0,
    confidence: torch.Tensor | None = None,
    penalize_negative_advantage: bool = False,
    zero_adv_inject_beta: float = 0.0,
    span_norm_ref_tokens: float = 0.0,
) -> C3Result:
    """Apply actor-only C3 modulation.

    For a bad event with severity ``q`` and confidence ``c``, a positive
    environment advantage ``A`` becomes::

        A_actor = -(1 + q * (reversal_scale + confidence_reversal_scale * c)) * A

    Clean tokens, non-positive advantages, and inactive loss-mask tokens are
    unchanged by default.  Overlapping bad events use the maximum severity per
    token, making the result independent of event order.  The returned tensor
    is a new tensor; ``advantage`` is never modified.
    """

    for name, value in {
        "reversal_scale": reversal_scale,
        "confidence_reversal_scale": confidence_reversal_scale,
        "zero_adv_inject_beta": zero_adv_inject_beta,
        "span_norm_ref_tokens": span_norm_ref_tokens,
    }.items():
        if float(value) < 0:
            raise ValueError(f"{name} must be non-negative")

    original = _validate_vector("advantage", advantage).detach().clone()
    length = original.numel()
    active = _validate_vector("loss_mask", loss_mask, length).to(device=original.device, dtype=torch.bool)
    confidence_values = None
    if confidence is not None:
        confidence_values = _validate_vector("confidence", confidence, length).detach().to(
            device=original.device, dtype=torch.float32
        ).clamp_(0.0, 1.0)

    bad = torch.zeros(length, dtype=torch.bool, device=original.device)
    severity = torch.zeros(length, dtype=torch.float32, device=original.device)
    valid_events: list[tuple[Mapping[str, Any], float, slice]] = []
    skipped_positive = 0.0
    pattern_counts: dict[str, float] = {}

    for event in events:
        value = _as_float(event.get("value"))
        if value is None or value == 0.0:
            continue
        if value > 0:
            skipped_positive += 1.0
            continue
        span = _event_slice(event, length)
        if span is None:
            continue
        q = min(abs(value), 1.0)
        bad[span] = True
        severity[span] = torch.maximum(severity[span], severity.new_full(severity[span].shape, q))
        valid_events.append((event, q, span))
        pattern = str(event.get("pattern") or "unknown")
        pattern_counts[pattern] = pattern_counts.get(pattern, 0.0) + 1.0

    active_bad = bad & active
    positive = original.clamp_min(0.0)
    negative = original.clamp_max(0.0)
    positive_conflict = active_bad & (original > 0)
    negative_conflict = active_bad & (original < 0)

    span_norm = torch.ones_like(original)
    clipped_count = 0.0
    active_span_sum = 0.0
    if span_norm_ref_tokens > 0:
        for _event, _q, span in valid_events:
            span_len = float(active[span].sum().item())
            active_span_sum += span_len
            if span_len > span_norm_ref_tokens:
                span_norm[span] = torch.minimum(
                    span_norm[span], span_norm.new_full(span_norm[span].shape, span_norm_ref_tokens / span_len)
                )
                clipped_count += 1.0

    confidence_scale = (
        confidence_values
        if confidence_values is not None
        else torch.zeros_like(original, dtype=torch.float32)
    )
    effective = severity.to(original.dtype) * (
        float(reversal_scale) + float(confidence_reversal_scale) * confidence_scale.to(original.dtype)
    ) * span_norm
    delta = torch.zeros_like(original)
    delta[positive_conflict] = -(1 + effective[positive_conflict]) * positive[positive_conflict]
    if penalize_negative_advantage:
        delta[negative_conflict] = effective[negative_conflict] * negative[negative_conflict]
    if zero_adv_inject_beta > 0:
        zero = active_bad & (original == 0)
        delta[zero] = -float(zero_adv_inject_beta) * severity.to(original.dtype)[zero] * confidence_scale.to(
            original.dtype
        )[zero] * span_norm[zero]

    result = original + delta
    applied_events: list[dict[str, Any]] = []
    for event, q, span in valid_events:
        mask = active[span]
        count = float(mask.sum().item())
        if count:
            weight = mask.to(original.dtype)
            before = float((original[span] * weight).sum().item() / count)
            after = float((result[span] * weight).sum().item() / count)
            event_delta = float((delta[span] * weight).sum().item() / count)
            conf_mean = float((confidence_scale[span] * mask.float()).sum().item() / count)
        else:
            before = after = event_delta = conf_mean = 0.0
        applied_events.append(
            {
                "pattern": str(event.get("pattern") or "unknown"),
                "value": float(event.get("value")),
                "start": int(span.start or 0),
                "end": int(span.stop or length),
                "severity": q,
                "confidence": conf_mean,
                "conflict_token_count": float((positive_conflict[span]).sum().item()),
                "advantage_mean_before": before,
                "advantage_mean_after": after,
                "actor_delta_mean": event_delta,
            }
        )

    positive_weight = positive_conflict.to(original.dtype)
    negative_weight = negative_conflict.to(original.dtype)
    metrics = {
        "event_count": float(len(valid_events)),
        "skipped_positive_event_count": skipped_positive,
        "bad_token_count": float(active_bad.sum().item()),
        "conflict_token_count": float(positive_conflict.sum().item()),
        "negative_conflict_token_count": float(negative_conflict.sum().item()),
        "positive_advantage_mass": float((positive * positive_weight).sum().item()),
        "actor_delta_sum": float((delta * active_bad.to(original.dtype)).sum().item()),
        "actor_delta_abs_sum": float((delta.abs() * active_bad.to(original.dtype)).sum().item()),
        "reversal_mass": float((effective * positive * positive_weight).sum().item()),
        "confidence_mean": float(
            confidence_scale[positive_conflict].mean().item()
            if confidence_values is not None and positive_conflict.any()
            else (1.0 if confidence_reversal_scale == 0 else 0.0)
        ),
        "span_norm_ref_tokens": float(span_norm_ref_tokens),
        "span_norm_clipped_event_count": clipped_count,
        "span_active_token_sum": active_span_sum,
    }
    metrics.update({f"pattern/{key}": value for key, value in pattern_counts.items()})
    return C3Result(advantages=result, metrics=metrics, applied_events=applied_events)
