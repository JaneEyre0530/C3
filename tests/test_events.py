import pytest

from c3 import detect_events, normalize_query, trigram_jaccard


def test_query_normalization_is_unicode_and_punctuation_stable():
    assert normalize_query("  Cafe\u0301\u200b,  Books! ") == "café books"
    assert trigram_jaccard("same query", "same query") == 1.0


def test_detector_emits_paper_events_and_action_confidence_span():
    turns = [
        {"action": "search", "query": "red shoes"},
        {"action": "search", "query": "red-shoes"},
        {"action": "answer"},
        {"action": "search", "query": "red boots"},
    ]
    events = detect_events(
        turns,
        [9, 19, 29, 39],
        action_spans=[{"start": 4, "end": 8}, {"start": 14, "end": 18}, None, {"start": 34, "end": 38}],
    )
    assert [event["pattern"] for event in events] == [
        "query_loop_exact",
        "invalid_action_after_answer",
    ]
    assert events[0]["target"] == {"start": 10, "end": 20}
    assert events[0]["confidence_target"] == {"start": 14, "end": 18}
    assert events[1]["value"] == -1.0

def test_detector_distinguishes_near_loops_and_ignores_strict_only_failure():
    events = detect_events(
        [
            {"action": "search", "query": "best noise cancelling wireless headphones for travel", "parse_ok": True},
            {"action": "search", "query": "best noise cancelling wireless headphones for travell", "parse_ok": True, "strict_parse_ok": False},
        ],
        [20, 40],
    )
    assert len(events) == 1
    assert events[0]["pattern"] == "query_loop_near"
    assert events[0]["value"] == -0.5


def test_detector_validates_parallel_inputs():
    with pytest.raises(ValueError, match="equal lengths"):
        detect_events([], [0])


def test_detector_accepts_serialized_parse_flags():
    events = detect_events([{"action": "search", "parse_ok": "0"}], [3])
    assert events[0]["pattern"] == "invalid_action"
