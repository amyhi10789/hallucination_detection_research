"""Prediction metrics and paired paraphrase-change evaluation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


def expected_calibration_error(
    y_true: Sequence[int | bool],
    y_probability: Sequence[float],
    *,
    n_bins: int = 10,
) -> float:
    """Return equal-width expected calibration error (ECE)."""

    labels = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(y_probability, dtype=float)
    if len(labels) != len(probabilities):
        raise ValueError("Labels and probabilities have different lengths.")
    if len(labels) == 0:
        return float("nan")
    if n_bins <= 0:
        raise ValueError("n_bins must be positive.")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Probabilities must be finite values in [0, 1].")

    # Include probability 1.0 in the final bin.
    bin_index = np.minimum((probabilities * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for index in range(n_bins):
        mask = bin_index == index
        if mask.any():
            ece += mask.mean() * abs(labels[mask].mean() - probabilities[mask].mean())
    return float(ece)


def classification_metrics(
    y_true: Sequence[int | bool],
    y_probability: Sequence[float],
    *,
    threshold: float = 0.5,
    n_bins: int = 10,
) -> dict[str, float | int]:
    """Compute discrimination, calibration, and threshold metrics safely."""

    labels = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(y_probability, dtype=float)
    if len(labels) != len(probabilities):
        raise ValueError("Labels and probabilities have different lengths.")
    if len(labels) == 0:
        return {
            "n": 0,
            "n_errors": 0,
            "error_rate": float("nan"),
            "mean_error_probability": float("nan"),
            "auroc": float("nan"),
            "auprc": float("nan"),
            "brier": float("nan"),
            "log_loss": float("nan"),
            "ece": float("nan"),
            "accuracy": float("nan"),
            "balanced_accuracy": float("nan"),
        }
    if not set(np.unique(labels).tolist()).issubset({0, 1}):
        raise ValueError("Labels must be binary (0=correct, 1=error).")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Probabilities must be finite values in [0, 1].")

    predicted = (probabilities >= threshold).astype(int)
    two_classes = np.unique(labels).size == 2
    clipped = np.clip(probabilities, np.finfo(float).eps, 1 - np.finfo(float).eps)
    return {
        "n": int(len(labels)),
        "n_errors": int(labels.sum()),
        "error_rate": float(labels.mean()),
        "mean_error_probability": float(probabilities.mean()),
        "auroc": float(roc_auc_score(labels, probabilities)) if two_classes else float("nan"),
        "auprc": (
            float(average_precision_score(labels, probabilities))
            if two_classes
            else float("nan")
        ),
        "brier": float(brier_score_loss(labels, probabilities)),
        "log_loss": float(log_loss(labels, clipped, labels=[0, 1])),
        "ece": expected_calibration_error(labels, probabilities, n_bins=n_bins),
        "accuracy": float(accuracy_score(labels, predicted)),
        "balanced_accuracy": (
            float(balanced_accuracy_score(labels, predicted))
            if two_classes
            else float("nan")
        ),
    }


def evaluate_by_groups(
    predictions: pd.DataFrame,
    *,
    group_cols: Sequence[str],
    label_col: str = "is_error",
    probability_col: str = "error_probability",
    threshold: float = 0.5,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Compute :func:`classification_metrics` within arbitrary groups."""

    required = [*group_cols, label_col, probability_col]
    missing = [column for column in required if column not in predictions]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    records: list[dict[str, Any]] = []
    if group_cols:
        grouper: str | list[str] = list(group_cols)
        grouped: Iterable[tuple[Any, pd.DataFrame]] = predictions.groupby(
            grouper, dropna=False, sort=False
        )
    else:
        grouped = [((), predictions)]
    for key, group in grouped:
        keys = key if isinstance(key, tuple) else (key,)
        record = dict(zip(group_cols, keys, strict=True))
        record.update(
            classification_metrics(
                group[label_col],
                group[probability_col],
                threshold=threshold,
                n_bins=n_bins,
            )
        )
        records.append(record)
    return pd.DataFrame.from_records(records)


def evaluate_standard_slices(
    predictions: pd.DataFrame,
    *,
    label_col: str = "is_error",
    probability_col: str = "error_probability",
    threshold: float = 0.5,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Evaluate overall, variant, subject, and subject-by-variant slices.

    Any available identifiers among ``model``, ``seed``, ``layer``, ``method``,
    and ``split`` are retained in every slice.
    """

    base = [
        column
        for column in ("model", "seed", "layer", "method", "split")
        if column in predictions
    ]
    scopes = {
        "overall": [],
        "variant": ["variant"],
        "subject": ["subject"],
        "subject_variant": ["subject", "variant"],
    }
    outputs: list[pd.DataFrame] = []
    for scope, extras in scopes.items():
        if not all(column in predictions for column in extras):
            continue
        result = evaluate_by_groups(
            predictions,
            group_cols=[*base, *extras],
            label_col=label_col,
            probability_col=probability_col,
            threshold=threshold,
            n_bins=n_bins,
        )
        result.insert(0, "scope", scope)
        outputs.append(result)
    return pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()


def _context_columns(frame: pd.DataFrame, requested: Sequence[str] | None) -> list[str]:
    if requested is not None:
        missing = [column for column in requested if column not in frame]
        if missing:
            raise KeyError(f"Missing context columns: {missing}")
        return list(requested)
    return [
        column
        for column in ("model", "seed", "layer", "method", "split", "subject")
        if column in frame
    ]


def paired_prediction_changes(
    predictions: pd.DataFrame,
    *,
    canonical_variant: str = "canonical",
    question_col: str = "question_id",
    variant_col: str = "variant",
    label_col: str = "is_error",
    probability_col: str = "error_probability",
    context_cols: Sequence[str] | None = None,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Pair each paraphrase with its canonical version for the same question."""

    required = [question_col, variant_col, label_col, probability_col]
    missing = [column for column in required if column not in predictions]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    context = _context_columns(predictions, context_cols)
    identity = [*context, question_col, variant_col]
    if predictions.duplicated(identity).any():
        raise ValueError(
            "Predictions contain duplicate context/question/variant rows; add the "
            "missing run identifiers to context_cols."
        )

    canonical = predictions[predictions[variant_col] == canonical_variant].copy()
    variants = predictions[predictions[variant_col] != canonical_variant].copy()
    if canonical.empty:
        raise ValueError(f"No canonical rows found for {canonical_variant!r}.")
    if variants.empty:
        return pd.DataFrame()

    keys = [*context, question_col]
    canonical = canonical[
        [*keys, label_col, probability_col]
    ].rename(
        columns={
            label_col: "canonical_is_error",
            probability_col: "canonical_error_probability",
        }
    )
    variants = variants[
        [*keys, variant_col, label_col, probability_col]
    ].rename(
        columns={
            label_col: "variant_is_error",
            probability_col: "variant_error_probability",
        }
    )
    paired = variants.merge(canonical, on=keys, how="inner", validate="many_to_one")
    paired["error_change"] = (
        paired["variant_is_error"].astype(int)
        - paired["canonical_is_error"].astype(int)
    ).astype(float)
    paired["prediction_change"] = (
        paired["variant_error_probability"]
        - paired["canonical_error_probability"]
    )
    paired["absolute_prediction_change"] = paired["prediction_change"].abs()
    paired["outcome_changed"] = paired["error_change"] != 0
    paired["detector_flip"] = (
        paired["variant_error_probability"].ge(threshold)
        != paired["canonical_error_probability"].ge(threshold)
    )
    paired["change_direction_agrees"] = np.where(
        paired["outcome_changed"],
        paired["prediction_change"] * paired["error_change"] > 0,
        np.nan,
    )
    return paired


def summarize_paired_changes(
    paired: pd.DataFrame,
    *,
    group_cols: Sequence[str] | None = None,
    variant_col: str = "variant",
) -> pd.DataFrame:
    """Aggregate signed error and detector changes on the same probability scale."""

    required = [
        "canonical_is_error",
        "variant_is_error",
        "canonical_error_probability",
        "variant_error_probability",
        "error_change",
        "prediction_change",
        "absolute_prediction_change",
        "outcome_changed",
        "detector_flip",
    ]
    missing = [column for column in required if column not in paired]
    if missing:
        raise KeyError(f"Missing paired-change columns: {missing}")
    if group_cols is None:
        group_cols = [
            column
            for column in ("model", "seed", "layer", "method", "split", "subject", variant_col)
            if column in paired
        ]

    records: list[dict[str, Any]] = []
    grouped: Iterable[tuple[Any, pd.DataFrame]]
    if group_cols:
        grouped = paired.groupby(list(group_cols), dropna=False, sort=False)
    else:
        grouped = [((), paired)]
    for key, group in grouped:
        keys = key if isinstance(key, tuple) else (key,)
        delta_error = float(group["error_change"].mean())
        delta_prediction = float(group["prediction_change"].mean())
        stable = group.loc[~group["outcome_changed"], "absolute_prediction_change"]
        changed = group.loc[group["outcome_changed"], "change_direction_agrees"]
        record: dict[str, Any] = dict(zip(group_cols, keys, strict=True))
        record.update(
            {
                "n_questions": int(len(group)),
                "canonical_error_rate": float(group["canonical_is_error"].mean()),
                "variant_error_rate": float(group["variant_is_error"].mean()),
                "canonical_mean_prediction": float(
                    group["canonical_error_probability"].mean()
                ),
                "variant_mean_prediction": float(
                    group["variant_error_probability"].mean()
                ),
                "delta_error": delta_error,
                "delta_prediction": delta_prediction,
                # Absolute aliases are intentionally named for the reporting
                # layer; the signed deltas above remain the inferential values.
                "actual_error_rate_change": abs(delta_error),
                "detector_score_change": abs(delta_prediction),
                "tracking_mismatch": abs(delta_prediction - delta_error),
                "excess_volatility": abs(delta_prediction) - abs(delta_error),
                "mean_absolute_score_drift": float(
                    group["absolute_prediction_change"].mean()
                ),
                "outcome_flip_rate": float(group["outcome_changed"].mean()),
                "detector_flip_rate": float(group["detector_flip"].mean()),
                "stable_outcome_score_drift": (
                    float(stable.mean()) if len(stable) else float("nan")
                ),
                "direction_agreement_when_outcome_changes": (
                    float(changed.astype(float).mean()) if len(changed) else float("nan")
                ),
            }
        )
        records.append(record)
    return pd.DataFrame.from_records(records)


def paraphrase_change_metrics(
    predictions: pd.DataFrame,
    **pairing_kwargs: Any,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience wrapper returning question-level pairs and their summary."""

    paired = paired_prediction_changes(predictions, **pairing_kwargs)
    return paired, summarize_paired_changes(paired) if not paired.empty else pd.DataFrame()

