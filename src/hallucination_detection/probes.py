"""Logistic hidden-state probes used by the baseline and robust methods."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def feature_matrix(
    frame: pd.DataFrame,
    feature: str | Sequence[str] = "hidden_state",
) -> np.ndarray:
    """Convert an array-valued column or numeric columns into a 2-D matrix."""

    if isinstance(feature, str):
        if feature not in frame:
            raise KeyError(f"Missing feature column: {feature!r}")
        values = [np.asarray(value, dtype=float).reshape(-1) for value in frame[feature]]
        if not values:
            raise ValueError("Cannot build a feature matrix from an empty frame.")
        widths = {value.size for value in values}
        if len(widths) != 1:
            raise ValueError("All hidden-state vectors must have the same length.")
        matrix = np.stack(values)
    else:
        columns = list(feature)
        missing = [column for column in columns if column not in frame]
        if missing:
            raise KeyError(f"Missing feature columns: {missing}")
        matrix = frame[columns].to_numpy(dtype=float)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("Features must be a finite two-dimensional matrix.")
    return matrix


def _binary_labels(values: Sequence[int | bool] | np.ndarray) -> np.ndarray:
    labels = np.asarray(values).reshape(-1)
    if labels.size == 0:
        raise ValueError("Cannot fit a probe without labels.")
    unique = np.unique(labels)
    if not set(unique.tolist()).issubset({0, 1, False, True}):
        raise ValueError("Probe labels must be binary (0=correct, 1=error).")
    return labels.astype(int)


class ConstantProbabilityProbe(BaseEstimator, ClassifierMixin):
    """A two-column constant predictor for one-class training data."""

    def fit(self, X: np.ndarray, y: Sequence[int | bool]) -> "ConstantProbabilityProbe":
        matrix = np.asarray(X)
        labels = _binary_labels(y)
        if len(matrix) != len(labels):
            raise ValueError("X and y have different lengths.")
        self.error_probability_ = float(labels.mean())
        self.classes_ = np.array([0, 1], dtype=int)
        self.n_features_in_ = matrix.shape[1]
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        matrix = np.asarray(X)
        if matrix.ndim != 2:
            raise ValueError("X must be two-dimensional.")
        if not hasattr(self, "error_probability_"):
            raise RuntimeError("Probe has not been fitted.")
        error = np.full(len(matrix), self.error_probability_, dtype=float)
        return np.column_stack((1.0 - error, error))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def _fit_logistic_or_constant(
    X: np.ndarray,
    y: np.ndarray,
    *,
    scale: bool,
    C: float,
    class_weight: str | Mapping[int, float] | None,
    max_iter: int,
    seed: int,
) -> BaseEstimator:
    if np.unique(y).size < 2:
        return ConstantProbabilityProbe().fit(X, y)
    logistic = LogisticRegression(
        C=C,
        class_weight=class_weight,
        max_iter=max_iter,
        random_state=seed,
        solver="lbfgs",
    )
    if scale:
        estimator: BaseEstimator = Pipeline(
            [("scale", StandardScaler()), ("logistic", logistic)]
        )
    else:
        estimator = logistic
    estimator.fit(X, y)
    return estimator


class LogisticProbe(BaseEstimator, ClassifierMixin):
    """Standardized logistic probe with a safe one-class fallback."""

    def __init__(
        self,
        *,
        C: float = 1.0,
        class_weight: str | Mapping[int, float] | None = "balanced",
        max_iter: int = 2_000,
        seed: int = 42,
    ) -> None:
        self.C = C
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.seed = seed

    def fit(self, X: np.ndarray, y: Sequence[int | bool]) -> "LogisticProbe":
        matrix = np.asarray(X, dtype=float)
        labels = _binary_labels(y)
        if matrix.ndim != 2 or len(matrix) != len(labels):
            raise ValueError("X must be 2-D and have the same length as y.")
        self.estimator_ = _fit_logistic_or_constant(
            matrix,
            labels,
            scale=True,
            C=self.C,
            class_weight=self.class_weight,
            max_iter=self.max_iter,
            seed=self.seed,
        )
        self.classes_ = np.array([0, 1], dtype=int)
        self.n_features_in_ = matrix.shape[1]
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "estimator_"):
            raise RuntimeError("Probe has not been fitted.")
        return self.estimator_.predict_proba(np.asarray(X, dtype=float))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class InvariantProjectionProbe(BaseEstimator, ClassifierMixin):
    """Remove dominant within-question wording directions before probing.

    Training features are standardized first.  For every question with at
    least two variants, its standardized group mean is subtracted.  Right
    singular vectors of these deviations are treated as wording directions
    and projected out before fitting the error classifier.
    """

    def __init__(
        self,
        *,
        explained_variance: float = 0.90,
        max_components: int = 32,
        C: float = 1.0,
        class_weight: str | Mapping[int, float] | None = "balanced",
        max_iter: int = 2_000,
        seed: int = 42,
    ) -> None:
        self.explained_variance = explained_variance
        self.max_components = max_components
        self.C = C
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.seed = seed

    def fit(
        self,
        X: np.ndarray,
        y: Sequence[int | bool],
        *,
        question_ids: Sequence[Any],
    ) -> "InvariantProjectionProbe":
        matrix = np.asarray(X, dtype=float)
        labels = _binary_labels(y)
        groups = np.asarray(question_ids)
        if matrix.ndim != 2 or len(matrix) != len(labels) or len(groups) != len(labels):
            raise ValueError("X, y, and question_ids must have matching lengths.")
        if not 0 < self.explained_variance <= 1:
            raise ValueError("explained_variance must be in (0, 1].")
        if self.max_components < 0:
            raise ValueError("max_components must be non-negative.")

        self.scaler_ = StandardScaler().fit(matrix)
        standardized = self.scaler_.transform(matrix)
        deviations: list[np.ndarray] = []
        for group in pd.unique(groups):
            group_matrix = standardized[groups == group]
            if len(group_matrix) >= 2:
                deviations.append(group_matrix - group_matrix.mean(axis=0, keepdims=True))

        components = np.empty((0, matrix.shape[1]), dtype=float)
        explained = np.empty(0, dtype=float)
        if deviations and self.max_components > 0:
            deviation_matrix = np.vstack(deviations)
            if np.linalg.norm(deviation_matrix) > np.finfo(float).eps:
                _, singular_values, right_vectors = np.linalg.svd(
                    deviation_matrix, full_matrices=False
                )
                variances = singular_values**2
                total = variances.sum()
                if total > 0:
                    ratios = variances / total
                    requested = int(
                        np.searchsorted(np.cumsum(ratios), self.explained_variance)
                        + 1
                    )
                    n_components = min(requested, self.max_components, len(right_vectors))
                    components = right_vectors[:n_components]
                    explained = ratios[:n_components]

        self.wording_components_ = components
        self.wording_explained_variance_ratio_ = explained
        projected = self._project_standardized(standardized)
        self.estimator_ = _fit_logistic_or_constant(
            projected,
            labels,
            scale=False,
            C=self.C,
            class_weight=self.class_weight,
            max_iter=self.max_iter,
            seed=self.seed,
        )
        self.classes_ = np.array([0, 1], dtype=int)
        self.n_features_in_ = matrix.shape[1]
        return self

    def _project_standardized(self, standardized: np.ndarray) -> np.ndarray:
        components = self.wording_components_
        if components.size == 0:
            return standardized
        return standardized - (standardized @ components.T) @ components

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "scaler_"):
            raise RuntimeError("Probe has not been fitted.")
        standardized = self.scaler_.transform(np.asarray(X, dtype=float))
        return self._project_standardized(standardized)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "estimator_"):
            raise RuntimeError("Probe has not been fitted.")
        return self.estimator_.predict_proba(self.transform(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def fit_probe_suite(
    train_frame: pd.DataFrame,
    *,
    feature: str | Sequence[str] = "hidden_state",
    label_col: str = "is_error",
    question_col: str = "question_id",
    variant_col: str = "variant",
    canonical_variant: str = "canonical",
    explained_variance: float = 0.90,
    max_components: int = 32,
    C: float = 1.0,
    class_weight: str | Mapping[int, float] | None = "balanced",
    max_iter: int = 2_000,
    seed: int = 42,
) -> dict[str, BaseEstimator]:
    """Fit canonical, augmented, and invariant logistic probes.

    The canonical baseline sees canonical rows only.  The augmented baseline
    and invariant method see every training variant.  The caller should pass
    only rows from the training split.
    """

    required = [label_col, question_col, variant_col]
    missing = [column for column in required if column not in train_frame]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    canonical = train_frame[train_frame[variant_col] == canonical_variant]
    if canonical.empty:
        raise ValueError(f"No rows found for canonical variant {canonical_variant!r}.")

    common = dict(C=C, class_weight=class_weight, max_iter=max_iter, seed=seed)
    baseline = LogisticProbe(**common).fit(
        feature_matrix(canonical, feature), canonical[label_col]
    )
    all_features = feature_matrix(train_frame, feature)
    augmented = LogisticProbe(**common).fit(all_features, train_frame[label_col])
    invariant = InvariantProjectionProbe(
        explained_variance=explained_variance,
        max_components=max_components,
        **common,
    ).fit(
        all_features,
        train_frame[label_col],
        question_ids=train_frame[question_col],
    )
    return {
        "canonical_logistic": baseline,
        "augmented_logistic": augmented,
        "invariant_projection": invariant,
    }


def predict_probe_suite(
    frame: pd.DataFrame,
    probes: Mapping[str, BaseEstimator],
    *,
    feature: str | Sequence[str] = "hidden_state",
    method_col: str = "method",
    probability_col: str = "error_probability",
) -> pd.DataFrame:
    """Return long-form predictions from every fitted probe."""

    matrix = feature_matrix(frame, feature)
    outputs: list[pd.DataFrame] = []
    for method, probe in probes.items():
        predicted = frame.copy()
        predicted[method_col] = method
        predicted[probability_col] = probe.predict_proba(matrix)[:, 1]
        outputs.append(predicted)
    if not outputs:
        return frame.iloc[0:0].assign(
            **{method_col: pd.Series(dtype="object"), probability_col: pd.Series(dtype=float)}
        )
    return pd.concat(outputs, ignore_index=True)


def save_probe(probe: BaseEstimator, path: str | Path) -> Path:
    """Persist a fitted probe and return the resolved output path."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(probe, output)
    return output.resolve()


def load_probe(path: str | Path) -> BaseEstimator:
    """Load a probe previously written by :func:`save_probe`."""

    return joblib.load(Path(path))

