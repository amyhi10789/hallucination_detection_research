"""Question-cluster bootstrap inference for paraphrase-change experiments."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BootstrapInterval:
    """A scalar estimate and percentile cluster-bootstrap interval."""

    estimate: float
    ci_low: float
    ci_high: float
    confidence_level: float
    n_clusters: int
    n_resamples: int
    distribution: np.ndarray


def _validate_bootstrap_inputs(
    frame: pd.DataFrame,
    cluster_col: str,
    n_resamples: int,
    confidence_level: float,
) -> np.ndarray:
    if cluster_col not in frame:
        raise KeyError(f"Missing cluster column: {cluster_col!r}")
    if frame.empty:
        raise ValueError("Cannot bootstrap an empty frame.")
    if frame[cluster_col].isna().any():
        raise ValueError("Cluster IDs may not be missing.")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive.")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be in (0, 1).")
    return pd.unique(frame[cluster_col])


def _cluster_lookup(frame: pd.DataFrame, cluster_col: str) -> dict[Any, pd.DataFrame]:
    return {
        cluster: group
        for cluster, group in frame.groupby(cluster_col, sort=False, dropna=False)
    }


def _draw_cluster_sample(
    lookup: dict[Any, pd.DataFrame],
    cluster_ids: np.ndarray,
    rng: np.random.Generator,
) -> pd.DataFrame:
    selected = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
    # Concatenating each occurrence (rather than filtering with ``isin``) keeps
    # multiplicities from sampling clusters with replacement.
    return pd.concat([lookup[cluster] for cluster in selected], ignore_index=True)


def cluster_bootstrap_ci(
    frame: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], float],
    *,
    cluster_col: str = "question_id",
    n_resamples: int = 2_000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> BootstrapInterval:
    """Estimate a scalar statistic with a paired question-cluster bootstrap."""

    clusters = _validate_bootstrap_inputs(
        frame, cluster_col, n_resamples, confidence_level
    )
    estimate = float(statistic(frame))
    lookup = _cluster_lookup(frame, cluster_col)
    rng = np.random.default_rng(seed)
    draws = np.array(
        [
            float(statistic(_draw_cluster_sample(lookup, clusters, rng)))
            for _ in range(n_resamples)
        ],
        dtype=float,
    )
    finite = draws[np.isfinite(draws)]
    if finite.size == 0:
        low = high = float("nan")
    else:
        alpha = 1.0 - confidence_level
        low, high = np.quantile(finite, (alpha / 2.0, 1.0 - alpha / 2.0))
    return BootstrapInterval(
        estimate=estimate,
        ci_low=float(low),
        ci_high=float(high),
        confidence_level=confidence_level,
        n_clusters=len(clusters),
        n_resamples=n_resamples,
        distribution=draws,
    )


def _change_statistics(frame: pd.DataFrame) -> dict[str, float]:
    required = ["error_change", "prediction_change"]
    missing = [column for column in required if column not in frame]
    if missing:
        raise KeyError(f"Missing paired-change columns: {missing}")
    delta_error = float(frame["error_change"].mean())
    delta_prediction = float(frame["prediction_change"].mean())
    absolute_change = (
        frame["absolute_prediction_change"]
        if "absolute_prediction_change" in frame
        else frame["prediction_change"].abs()
    )
    detector_flip = (
        float(frame["detector_flip"].mean())
        if "detector_flip" in frame
        else float("nan")
    )
    return {
        "delta_error": delta_error,
        "delta_prediction": delta_prediction,
        "actual_error_rate_change": abs(delta_error),
        "detector_score_change": abs(delta_prediction),
        "tracking_mismatch": abs(delta_prediction - delta_error),
        "excess_volatility": abs(delta_prediction) - abs(delta_error),
        "mean_absolute_score_drift": float(absolute_change.mean()),
        "detector_flip_rate": detector_flip,
    }


def bootstrap_change_statistics(
    paired: pd.DataFrame,
    *,
    group_cols: Sequence[str],
    cluster_col: str = "question_id",
    n_resamples: int = 2_000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> pd.DataFrame:
    """Return estimates and CIs for every paired-change metric by contrast."""

    required = [*group_cols, cluster_col, "error_change", "prediction_change"]
    missing = [column for column in required if column not in paired]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    records: list[dict[str, Any]] = []
    grouped: Iterable[tuple[Any, pd.DataFrame]]
    if group_cols:
        grouped = paired.groupby(list(group_cols), sort=False, dropna=False)
    else:
        grouped = [((), paired)]

    for group_index, (key, group) in enumerate(grouped):
        keys = key if isinstance(key, tuple) else (key,)
        clusters = _validate_bootstrap_inputs(
            group, cluster_col, n_resamples, confidence_level
        )
        lookup = _cluster_lookup(group, cluster_col)
        rng = np.random.default_rng(seed + group_index)
        estimate = _change_statistics(group)
        bootstrap = {name: np.empty(n_resamples) for name in estimate}
        for draw_index in range(n_resamples):
            sample = _draw_cluster_sample(lookup, clusters, rng)
            values = _change_statistics(sample)
            for name, value in values.items():
                bootstrap[name][draw_index] = value

        alpha = 1.0 - confidence_level
        record: dict[str, Any] = dict(zip(group_cols, keys, strict=True))
        record.update(n_questions=int(group[cluster_col].nunique()))
        for name, value in estimate.items():
            finite = bootstrap[name][np.isfinite(bootstrap[name])]
            if finite.size:
                low, high = np.quantile(finite, (alpha / 2.0, 1.0 - alpha / 2.0))
            else:
                low = high = float("nan")
            record[name] = value
            record[f"{name}_ci_low"] = float(low)
            record[f"{name}_ci_high"] = float(high)
        records.append(record)
    return pd.DataFrame.from_records(records)


def _mean_contrast_excess(
    frame: pd.DataFrame,
    contrast_cols: Sequence[str],
) -> float:
    if contrast_cols:
        values = [
            _change_statistics(group)["excess_volatility"]
            for _, group in frame.groupby(list(contrast_cols), sort=False, dropna=False)
        ]
    else:
        values = [_change_statistics(frame)["excess_volatility"]]
    return float(np.mean(values))


def directional_excess_volatility_test(
    paired: pd.DataFrame,
    *,
    contrast_cols: Sequence[str] = ("subject", "variant"),
    cluster_col: str = "question_id",
    n_resamples: int = 2_000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict[str, float | int | bool]:
    """Test the directional hypothesis that detector changes are excessive.

    The reported one-sided lower bound is the ``1-confidence_level`` bootstrap
    quantile.  ``bootstrap_probability_nonpositive`` is a descriptive tail
    probability, while ``supports_hypothesis`` is true only when that lower
    bound is above zero.

    The input should normally be filtered to one model/layer/method/split.  A
    contrast is then a subject-by-paraphrase comparison by default.
    """

    missing = [
        column
        for column in [*contrast_cols, cluster_col, "error_change", "prediction_change"]
        if column not in paired
    ]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    clusters = _validate_bootstrap_inputs(
        paired, cluster_col, n_resamples, confidence_level
    )
    lookup = _cluster_lookup(paired, cluster_col)
    statistic = lambda sample: _mean_contrast_excess(sample, contrast_cols)
    estimate = statistic(paired)
    rng = np.random.default_rng(seed)
    distribution = np.array(
        [
            statistic(_draw_cluster_sample(lookup, clusters, rng))
            for _ in range(n_resamples)
        ]
    )
    finite = distribution[np.isfinite(distribution)]
    if finite.size:
        alpha = 1.0 - confidence_level
        lower_one_sided = float(np.quantile(finite, alpha))
        two_sided_low, two_sided_high = np.quantile(
            finite, (alpha / 2.0, 1.0 - alpha / 2.0)
        )
        nonpositive = float((np.count_nonzero(finite <= 0) + 1) / (len(finite) + 1))
    else:
        lower_one_sided = two_sided_low = two_sided_high = nonpositive = float("nan")
    n_contrasts = (
        paired.groupby(list(contrast_cols), dropna=False).ngroups
        if contrast_cols
        else 1
    )
    return {
        "mean_excess_volatility": float(estimate),
        "ci_low": float(two_sided_low),
        "ci_high": float(two_sided_high),
        "one_sided_lower_bound": lower_one_sided,
        "bootstrap_probability_nonpositive": nonpositive,
        "supports_hypothesis": bool(lower_one_sided > 0),
        "confidence_level": confidence_level,
        "n_questions": int(len(clusters)),
        "n_contrasts": int(n_contrasts),
        "n_resamples": int(n_resamples),
    }


def stable_margin_analysis(
    paired: pd.DataFrame,
    *,
    contrast_cols: Sequence[str],
    margin: float = 0.05,
    cluster_col: str = "question_id",
    n_resamples: int = 2_000,
    equivalence_confidence: float = 0.90,
    seed: int = 42,
) -> pd.DataFrame:
    """Identify contrasts whose actual error-rate change is equivalent to zero.

    A contrast is stable only when its paired bootstrap interval for
    ``delta_error`` lies wholly inside ``[-margin, +margin]``.  A 90% interval
    corresponds to the usual two-one-sided-tests convention at alpha 0.05.
    """

    if margin <= 0 or margin >= 1:
        raise ValueError("margin must be in (0, 1).")
    required = [*contrast_cols, cluster_col, "error_change", "prediction_change"]
    missing = [column for column in required if column not in paired]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    records: list[dict[str, Any]] = []
    grouped: Iterable[tuple[Any, pd.DataFrame]]
    if contrast_cols:
        grouped = paired.groupby(list(contrast_cols), sort=False, dropna=False)
    else:
        grouped = [((), paired)]
    alpha = 1.0 - equivalence_confidence

    for group_index, (key, group) in enumerate(grouped):
        keys = key if isinstance(key, tuple) else (key,)
        clusters = _validate_bootstrap_inputs(
            group, cluster_col, n_resamples, equivalence_confidence
        )
        lookup = _cluster_lookup(group, cluster_col)
        rng = np.random.default_rng(seed + group_index)
        delta_errors = np.empty(n_resamples)
        delta_predictions = np.empty(n_resamples)
        for index in range(n_resamples):
            sample = _draw_cluster_sample(lookup, clusters, rng)
            delta_errors[index] = sample["error_change"].mean()
            delta_predictions[index] = sample["prediction_change"].mean()
        error_low, error_high = np.quantile(
            delta_errors, (alpha / 2.0, 1.0 - alpha / 2.0)
        )
        prediction_low, prediction_high = np.quantile(
            delta_predictions, (alpha / 2.0, 1.0 - alpha / 2.0)
        )
        delta_error = float(group["error_change"].mean())
        delta_prediction = float(group["prediction_change"].mean())
        stable = bool(error_low > -margin and error_high < margin)
        stable_outcomes = group[group["error_change"] == 0]
        stable_drift = (
            float(stable_outcomes["prediction_change"].abs().mean())
            if len(stable_outcomes)
            else float("nan")
        )
        record: dict[str, Any] = dict(zip(contrast_cols, keys, strict=True))
        record.update(
            {
                "n_questions": int(len(clusters)),
                "equivalence_margin": margin,
                "delta_error": delta_error,
                "delta_error_ci_low": float(error_low),
                "delta_error_ci_high": float(error_high),
                "delta_prediction": delta_prediction,
                "delta_prediction_ci_low": float(prediction_low),
                "delta_prediction_ci_high": float(prediction_high),
                "is_error_rate_stable": stable,
                "absolute_detector_change": abs(delta_prediction),
                "detector_change_exceeds_margin": abs(delta_prediction) > margin,
                "stable_outcome_score_drift": stable_drift,
            }
        )
        records.append(record)
    return pd.DataFrame.from_records(records)

