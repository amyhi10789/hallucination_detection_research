"""Canonical tabular schemas shared by the experiment pipeline."""

from __future__ import annotations

import ast
import json
from typing import Any

import pandas as pd


class SchemaError(ValueError):
    """Raised when a pipeline table violates its expected schema."""


QUESTION_COLUMNS = (
    "question_id",
    "source",
    "source_split",
    "subject",
    "question",
    "choices",
    "answer_index",
)

VARIANT_COLUMNS = (
    "variant_id",
    "question_id",
    "subject",
    "question_text",
    "choices",
    "answer_index",
    "variant",
    "semantic_status",
    "reviewer_notes",
    "source",
)

SEMANTIC_STATUSES = frozenset({"canonical", "validated", "unreviewed", "rejected"})


def coerce_choices(value: Any) -> tuple[str, str, str, str]:
    """Normalize a four-choice value from datasets, JSON, or CSV."""

    parsed = value
    if isinstance(value, str):
        text = value.strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(text)
            except (SyntaxError, ValueError) as exc:
                raise SchemaError("choices must be a four-item sequence") from exc
    if not isinstance(parsed, (list, tuple)) or len(parsed) != 4:
        raise SchemaError("choices must contain exactly four options")
    choices = tuple(str(item).strip() for item in parsed)
    if any(not item for item in choices):
        raise SchemaError("choices cannot contain blank options")
    return choices  # type: ignore[return-value]


def coerce_answer_index(value: Any) -> int:
    """Normalize an answer represented as a zero-based index or A-D letter."""

    if isinstance(value, str):
        cleaned = value.strip().upper()
        if cleaned in {"A", "B", "C", "D"}:
            return ord(cleaned) - ord("A")
        try:
            value = int(cleaned)
        except ValueError as exc:
            raise SchemaError(f"Invalid answer value: {value!r}") from exc
    try:
        answer = int(value)
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"Invalid answer value: {value!r}") from exc
    if answer not in range(4):
        raise SchemaError("answer_index must be between 0 and 3")
    return answer


def _require_columns(frame: pd.DataFrame, required: tuple[str, ...], name: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise SchemaError(f"{name} is missing columns: {', '.join(missing)}")


def validate_questions(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a normalized copy of a canonical question table."""

    _require_columns(frame, QUESTION_COLUMNS, "question table")
    if frame.empty:
        raise SchemaError("question table is empty")
    result = frame.loc[:, QUESTION_COLUMNS].copy()
    for column in ("question_id", "source", "source_split", "subject", "question"):
        result[column] = result[column].map(lambda value: str(value).strip())
        if (result[column] == "").any():
            raise SchemaError(f"question table contains blank {column} values")
    result["choices"] = result["choices"].map(coerce_choices)
    result["answer_index"] = result["answer_index"].map(coerce_answer_index)
    if result["question_id"].duplicated().any():
        duplicates = result.loc[result["question_id"].duplicated(), "question_id"].tolist()
        raise SchemaError(f"question_id values must be unique; duplicates: {duplicates[:3]}")
    return result.reset_index(drop=True)


def validate_variants(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a normalized copy of a prompt-variant table."""

    _require_columns(frame, VARIANT_COLUMNS, "variant table")
    if frame.empty:
        raise SchemaError("variant table is empty")
    result = frame.loc[:, VARIANT_COLUMNS].copy()
    text_columns = (
        "variant_id",
        "question_id",
        "subject",
        "question_text",
        "variant",
        "semantic_status",
        "source",
    )
    for column in text_columns:
        result[column] = result[column].map(lambda value: str(value).strip())
        if (result[column] == "").any():
            raise SchemaError(f"variant table contains blank {column} values")
    result["reviewer_notes"] = result["reviewer_notes"].fillna("").map(
        lambda value: str(value).strip()
    )
    result["choices"] = result["choices"].map(coerce_choices)
    result["answer_index"] = result["answer_index"].map(coerce_answer_index)
    if result["variant_id"].duplicated().any():
        raise SchemaError("variant_id values must be unique")
    invalid_statuses = sorted(set(result["semantic_status"]) - SEMANTIC_STATUSES)
    if invalid_statuses:
        raise SchemaError(f"Unknown semantic_status values: {invalid_statuses}")
    return result.reset_index(drop=True)
