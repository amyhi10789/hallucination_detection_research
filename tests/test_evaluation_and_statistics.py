import pandas as pd

from hallucination_detection.evaluation import (
    classification_metrics,
    paired_prediction_changes,
    summarize_paired_changes,
)
from hallucination_detection.statistics import directional_excess_volatility_test


def _predictions():
    rows = []
    for index in range(12):
        base_error = index % 3 == 0
        for variant in ("canonical", "direct"):
            rows.append(
                {
                    "question_id": f"q{index}",
                    "subject": "science" if index < 6 else "history",
                    "variant": variant,
                    "is_error": base_error,
                    "error_probability": (0.8 if base_error else 0.2)
                    + (0.12 if variant == "direct" else 0.0),
                    "method": "canonical_logistic",
                    "layer": 4,
                    "split": "test",
                }
            )
    return pd.DataFrame(rows)


def test_change_analysis_matches_stable_error_example():
    paired = paired_prediction_changes(_predictions())
    summary = summarize_paired_changes(paired, group_cols=["variant"])
    result = directional_excess_volatility_test(
        paired,
        contrast_cols=["subject", "variant"],
        n_resamples=20,
        seed=2,
    )

    assert summary.loc[0, "actual_error_rate_change"] == 0
    assert summary.loc[0, "detector_score_change"] > 0
    assert result["mean_excess_volatility"] > 0


def test_classification_metrics_handle_regular_binary_case():
    metrics = classification_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert metrics["auroc"] == 1.0
    assert metrics["brier"] < 0.05

