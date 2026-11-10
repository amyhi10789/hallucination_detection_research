from __future__ import annotations

import json

import pandas as pd
import pytest

from hallucination_detection.reporting import generate_report


def _synthetic_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions = pd.DataFrame(
        {
            "question_id": ["q1", "q1", "q2", "q2", "q3", "q3", "q4", "q4"],
            "subject": ["history"] * 4 + ["physics"] * 4,
            "variant": ["original", "paraphrase"] * 4,
            "layer": [6] * 8,
            "method": ["baseline"] * 8,
            "detector_score": [0.10, 0.32, 0.22, 0.55, 0.61, 0.72, 0.80, 0.94],
            "is_error": [0, 0, 0, 1, 1, 1, 1, 1],
            "parse_success": [True, True, True, False, True, True, True, True],
        }
    )
    tracking = pd.DataFrame(
        {
            "pair_id": ["q1", "q2", "q3", "q4"],
            "method": ["baseline", "baseline", "consistent", "consistent"],
            "actual_error_rate_change": [0.00, 0.10, 0.05, 0.10],
            "detector_score_change": [0.22, 0.33, 0.11, 0.14],
        }
    )
    layers = pd.DataFrame(
        {
            "layer": [2, 6, 10],
            "auroc": [0.55, 0.72, 0.67],
            "auprc": [0.58, 0.75, 0.70],
            "brier": [0.26, 0.19, 0.22],
            "n": [8, 8, 8],
        }
    )
    return predictions, tracking, layers


def test_generate_report_writes_complete_artifact_bundle(tmp_path):
    predictions, tracking, layers = _synthetic_frames()

    artifacts = generate_report(
        tmp_path / "smoke-run",
        profile="smoke",
        predictions=predictions,
        tracking=tracking,
        layer_metrics=layers,
        summary={"metrics": {"overall_auroc": 0.72}},
        metadata={"model": "synthetic/model", "seed": 17},
        limitations=["Synthetic test data are not research results."],
    )

    assert artifacts.report.is_file()
    assert artifacts.summary.is_file()
    assert set(artifacts.tables) == {
        "predictions",
        "tracking_pairs",
        "layer_metrics",
        "calibration",
    }
    assert set(artifacts.figures) == {
        "tracking_scatter",
        "layer_metrics",
        "detector_drift_by_method",
        "calibration",
    }
    assert all(path.stat().st_size > 0 for path in artifacts.tables.values())
    assert all(path.stat().st_size > 0 for path in artifacts.figures.values())

    summary = json.loads(artifacts.summary.read_text(encoding="utf-8"))
    assert summary["profile"] == "smoke"
    assert summary["sample_sizes"]["questions"] == 4
    assert summary["parse_failures"]["count"] == 1
    assert summary["hypothesis"]["mean_excess_detector_change"] > 0
    assert summary["hypothesis"]["status"] == "descriptively_consistent"

    report = artifacts.report.read_text(encoding="utf-8")
    assert "SMOKE" in report
    assert "Hypothesis Result" in report
    assert "Parse failures" in report
    assert "figures/tracking_scatter.png" in report
    assert "tables/predictions.csv" in report
    assert "Synthetic test data are not research results." in report


def test_generate_report_accepts_explicit_calibration_and_confirmatory_result(tmp_path):
    calibration = pd.DataFrame(
        {
            "mean_predicted_error": [0.2, 0.7],
            "observed_error_rate": [0.1, 0.8],
            "count": [20, 20],
        }
    )

    artifacts = generate_report(
        tmp_path / "full-run",
        profile="full",
        calibration=calibration,
        summary={
            "hypothesis": {
                "supports_hypothesis": True,
                "p_value": 0.012,
                "statement": "The preregistered paired test supports the hypothesis.",
            }
        },
    )

    assert set(artifacts.tables) == {"calibration"}
    assert set(artifacts.figures) == {"calibration"}
    payload = json.loads(artifacts.summary.read_text(encoding="utf-8"))
    assert payload["hypothesis"]["supported"] is True
    assert payload["hypothesis"]["status"] == "supported"
    assert "preregistered paired test" in artifacts.report.read_text(encoding="utf-8")


def test_generate_report_rejects_unknown_profile(tmp_path):
    with pytest.raises(ValueError, match="profile must be one of"):
        generate_report(tmp_path, profile="tiny")

