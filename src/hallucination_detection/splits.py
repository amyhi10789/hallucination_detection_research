"""Leakage-safe train/validation/test splitting utilities.

The original question, rather than an individual paraphrase, is the sampling
unit.  Every row sharing a ``question_id`` is therefore assigned to the same
split.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd


STANDARD_SPLITS = ("train", "validation", "test")


def _validate_fractions(fractions: Sequence[float]) -> np.ndarray:
    values = np.asarray(fractions, dtype=float)
    if values.shape != (3,):
        raise ValueError("Expected train, validation, and test fractions.")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Split fractions must be finite and non-negative.")
    if not np.isclose(values.sum(), 1.0):
        raise ValueError("Split fractions must sum to 1.0.")
    return values


def _largest_remainder_counts(n_items: int, fractions: np.ndarray) -> np.ndarray:
    """Allocate ``n_items`` deterministically while respecting fractions."""

    raw = fractions * n_items
    counts = np.floor(raw).astype(int)
    remainder = n_items - int(counts.sum())
    if remainder:
        order = np.argsort(-(raw - counts), kind="stable")
        counts[order[:remainder]] += 1

    # When possible, make every requested split non-empty.  This is helpful for
    # small subject strata and never changes the total number of questions.
    positive = np.flatnonzero(fractions > 0)
    if n_items >= len(positive):
        for destination in positive[counts[positive] == 0]:
            donors = positive[counts[positive] > 1]
            if len(donors):
                donor = donors[np.argmax(counts[donors])]
                counts[donor] -= 1
                counts[destination] += 1
    return counts


def split_by_question(
    frame: pd.DataFrame,
    *,
    question_col: str = "question_id",
    subject_col: str = "subject",
    split_col: str = "split",
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    heldout_subjects: Iterable[str] | None = None,
    heldout_split: str = "heldout",
    stratify_by_subject: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    """Return a copy with leakage-safe split assignments.

    Parameters
    ----------
    frame:
        One row per question variant (or any long-form table containing
        repeated question IDs).
    heldout_subjects:
        Subjects excluded from all ordinary splits and assigned to
        ``heldout_split``.  They are useful for a true cross-domain test.
    stratify_by_subject:
        Split question IDs separately inside each non-held-out subject.  Set to
        ``False`` to shuffle all remaining question IDs together.

    Notes
    -----
    A question ID must map to exactly one subject.  Splitting happens before
    merging the assignment back to every paraphrase row.
    """

    if question_col not in frame:
        raise KeyError(f"Missing question column: {question_col!r}")
    if frame[question_col].isna().any():
        raise ValueError(f"{question_col!r} may not contain missing values.")
    if stratify_by_subject or heldout_subjects:
        if subject_col not in frame:
            raise KeyError(f"Missing subject column: {subject_col!r}")

    fractions = _validate_fractions(
        (train_fraction, validation_fraction, test_fraction)
    )
    heldout = set(heldout_subjects or ())
    question_columns = [question_col]
    if subject_col in frame:
        question_columns.append(subject_col)
    questions = frame[question_columns].drop_duplicates().copy()

    if subject_col in questions:
        subject_counts = questions.groupby(question_col, dropna=False)[
            subject_col
        ].nunique(dropna=False)
        inconsistent = subject_counts[subject_counts > 1]
        if not inconsistent.empty:
            examples = inconsistent.index.astype(str).tolist()[:5]
            raise ValueError(
                "Each question must belong to one subject; inconsistent IDs: "
                + ", ".join(examples)
            )
        questions = questions.drop_duplicates(question_col)

    questions[split_col] = pd.Series(index=questions.index, dtype="object")
    if heldout:
        heldout_mask = questions[subject_col].isin(heldout)
        questions.loc[heldout_mask, split_col] = heldout_split

    remaining = questions[questions[split_col].isna()]
    if remaining.empty:
        result = frame.drop(columns=[split_col], errors="ignore").merge(
            questions[[question_col, split_col]], on=question_col, how="left"
        )
        assert_no_question_leakage(result, question_col=question_col, split_col=split_col)
        return result

    rng = np.random.default_rng(seed)
    if stratify_by_subject and subject_col in remaining:
        strata = [group for _, group in remaining.groupby(subject_col, dropna=False)]
    else:
        strata = [remaining]

    for stratum in strata:
        indices = stratum.index.to_numpy(copy=True)
        rng.shuffle(indices)
        counts = _largest_remainder_counts(len(indices), fractions)
        start = 0
        for name, count in zip(STANDARD_SPLITS, counts, strict=True):
            selected = indices[start : start + count]
            questions.loc[selected, split_col] = name
            start += count

    result = frame.drop(columns=[split_col], errors="ignore").merge(
        questions[[question_col, split_col]], on=question_col, how="left"
    )
    assert_no_question_leakage(result, question_col=question_col, split_col=split_col)
    return result


def assert_no_question_leakage(
    frame: pd.DataFrame,
    *,
    question_col: str = "question_id",
    split_col: str = "split",
) -> None:
    """Raise ``ValueError`` if a question appears in more than one split."""

    missing = [column for column in (question_col, split_col) if column not in frame]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    if frame[split_col].isna().any():
        raise ValueError(f"{split_col!r} contains unassigned rows.")
    split_counts = frame.groupby(question_col, dropna=False)[split_col].nunique()
    leaked = split_counts[split_counts > 1]
    if not leaked.empty:
        examples = leaked.index.astype(str).tolist()[:5]
        raise ValueError(
            "Paraphrase leakage detected for question IDs: " + ", ".join(examples)
        )


def split_summary(
    frame: pd.DataFrame,
    *,
    question_col: str = "question_id",
    split_col: str = "split",
    subject_col: str = "subject",
) -> pd.DataFrame:
    """Count independent questions and long-form rows in every split."""

    group_columns = [split_col]
    if subject_col in frame:
        group_columns.append(subject_col)
    return (
        frame.groupby(group_columns, dropna=False)
        .agg(n_questions=(question_col, "nunique"), n_rows=(question_col, "size"))
        .reset_index()
        .sort_values(group_columns, kind="stable")
        .reset_index(drop=True)
    )

