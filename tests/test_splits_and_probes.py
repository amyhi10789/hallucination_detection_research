import numpy as np
import pandas as pd

from hallucination_detection.probes import fit_probe_suite, predict_probe_suite
from hallucination_detection.splits import assert_no_question_leakage, split_by_question


def test_question_variants_never_cross_splits():
    frame = pd.DataFrame(
        [
            {"question_id": f"q{question}", "subject": f"s{question % 3}", "variant": variant}
            for question in range(30)
            for variant in ("canonical", "direct", "exam")
        ]
    )
    split = split_by_question(frame, seed=11, heldout_subjects=["s2"])

    assert_no_question_leakage(split)
    assert set(split.loc[split["subject"] == "s2", "split"]) == {"heldout"}
    assert {"train", "validation", "test"}.issubset(set(split["split"]))


def test_probe_suite_predicts_probabilities_for_all_methods():
    rows = []
    rng = np.random.default_rng(4)
    for question in range(30):
        is_error = question % 2
        for variant, wording_shift in (("canonical", 0.0), ("direct", 3.0), ("exam", -3.0)):
            feature = rng.normal(size=8)
            feature[0] += 2.5 if is_error else -2.5
            feature[1] += wording_shift
            rows.append(
                {
                    "question_id": f"q{question}",
                    "variant": variant,
                    "is_error": is_error,
                    "hidden_state": feature,
                }
            )
    frame = pd.DataFrame(rows)
    probes = fit_probe_suite(frame, max_components=2, seed=4)
    predictions = predict_probe_suite(frame, probes)

    assert set(predictions["method"]) == {
        "canonical_logistic",
        "augmented_logistic",
        "invariant_projection",
    }
    assert predictions["error_probability"].between(0, 1).all()

