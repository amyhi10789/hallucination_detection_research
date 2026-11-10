import pytest

from hallucination_detection.inference import parse_answer, resolve_layer_indices


CHOICES = ["Mercury", "Venus", "Earth", "Mars"]


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("A", 0),
        ("(B)", 1),
        ("C. Earth", 2),
        ("The answer is D.", 3),
        ("Option B is correct.", 1),
        ("Earth", 2),
    ],
)
def test_parse_answer_accepts_supported_forms(response, expected):
    parsed, reason = parse_answer(response, CHOICES)
    assert parsed == expected
    assert reason != "unparseable"


@pytest.mark.parametrize("response", ["", "I cannot answer", "A planet may be rocky"])
def test_parse_answer_keeps_invalid_responses_separate(response):
    parsed, _ = parse_answer(response, CHOICES)
    assert parsed is None


def test_resolve_layer_fractions_and_absolute_indices():
    assert resolve_layer_indices([0.25, 0.5, 1.0, 7], 24) == [6, 7, 12, 24]
    assert resolve_layer_indices([-1], 24) == [24]


def test_resolve_layer_rejects_out_of_range_values():
    with pytest.raises(ValueError):
        resolve_layer_indices([1.2], 24)

