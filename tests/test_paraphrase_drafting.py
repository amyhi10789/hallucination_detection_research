from hallucination_detection.paraphrase_drafting import anchor_warnings, clean_candidate


def test_clean_candidate_removes_model_prefix_and_extra_lines():
    assert clean_candidate('Paraphrased question: "Which planet is closest?"\nExtra') == (
        "Which planet is closest?"
    )


def test_anchor_warnings_flag_numbers_and_negation():
    warnings = anchor_warnings(
        "Which option is not true for 25% of cases?",
        "Which option is true for the cases?",
    )
    assert any("NUMBER" in warning for warning in warnings)
    assert any("LOGIC" in warning for warning in warnings)

