import pytest
import torch

from c3 import apply_c3, policy_confidence


def test_positive_advantage_is_reversed_with_max_overlap_severity():
    result = apply_c3(
        torch.tensor([1.0, 2.0, 3.0, -1.0]),
        torch.ones(4),
        [
            {"pattern": "invalid", "value": -0.5, "target": {"start": 0, "end": 3}},
            {"pattern": "loop", "value": -1.0, "target": {"start": 1, "end": 2}},
        ],
        reversal_scale=0.0,
        confidence_reversal_scale=1.0,
        confidence=torch.full((4,), 0.5),
    )
    # The overlap uses the maximum severity once, not an additive penalty.
    assert torch.allclose(result.advantages, torch.tensor([-0.25, -1.0, -0.75, -1.0]))
    assert result.metrics["conflict_token_count"] == 3


def test_clean_negative_and_masked_tokens_are_unchanged():
    advantage = torch.tensor([0.4, -0.2, 0.8, 0.7])
    result = apply_c3(
        advantage,
        torch.tensor([1, 1, 0, 1]),
        [{"value": -1.0, "target": {"start": 0, "end": 4}}],
        confidence=torch.ones(4),
    )
    assert torch.allclose(result.advantages, torch.tensor([-0.4, -0.2, 0.8, -0.7]))
    assert torch.equal(advantage, torch.tensor([0.4, -0.2, 0.8, 0.7]))


def test_optional_negative_penalty_and_zero_injection():
    result = apply_c3(
        torch.tensor([0.0, -1.0]),
        torch.ones(2),
        [{"value": -0.5, "target": {"start": 0, "end": 2}}],
        confidence=torch.ones(2),
        penalize_negative_advantage=True,
        zero_adv_inject_beta=0.2,
    )
    assert torch.allclose(result.advantages, torch.tensor([-0.1, -1.5]))


@pytest.mark.parametrize(
    ("mode", "expected"),
    [("token_prob", [0.25, 0.5]), ("span_prob", [0.353553, 0.353553]), ("span_mean_prob", [0.375, 0.375])],
)
def test_policy_confidence_modes(mode, expected):
    log_probs = torch.log(torch.tensor([0.25, 0.5]))
    confidence = policy_confidence(log_probs, torch.ones(2), [{"value": -1, "target": {"start": 0, "end": 2}}], mode=mode)
    assert torch.allclose(confidence, torch.tensor(expected), atol=1e-5)


def test_confidence_target_can_be_narrower_than_event():
    confidence = policy_confidence(
        torch.log(torch.tensor([0.25, 0.5, 0.75])),
        torch.ones(3),
        [{"value": -1, "target": {"start": 0, "end": 3}, "confidence_target": {"start": 1, "end": 3}}],
        mode="span_prob",
    )
    # The narrower span controls confidence, while the event span receives it.
    expected = (0.5 * 0.75) ** 0.5
    assert torch.allclose(confidence, torch.full((3,), expected), atol=1e-5)


def test_invalid_configuration_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        apply_c3(torch.ones(1), torch.ones(1), [], reversal_scale=-1)
    with pytest.raises(ValueError, match="confidence mode"):
        policy_confidence(torch.zeros(1), torch.ones(1), [], mode="unknown")
