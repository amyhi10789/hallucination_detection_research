"""Reproducible prompt wrappers and optional human-curated paraphrases."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from .config import ParaphraseConfig
from .schemas import SchemaError, VARIANT_COLUMNS, validate_questions, validate_variants


WRAPPER_TEMPLATES: dict[str, str] = {
    "canonical": "{question}",
    "direct": "Answer the following question.\n{question}",
    "exam": "Select the best answer to this question.\n{question}",
    "minimal": "Choose the correct option.\n{question}",
}


def _variant_id(question_id: str, variant_type: str, text: str) -> str:
    digest = hashlib.sha256(
        f"{question_id}\0{variant_type}\0{text}".encode("utf-8")
    ).hexdigest()[:16]
    return f"v_{digest}"


def _template_variants(
    questions: pd.DataFrame, template_names: tuple[str, ...], include_canonical: bool
) -> list[dict[str, object]]:
    names = list(template_names)
    if include_canonical and "canonical" not in names:
        names.insert(0, "canonical")
    if not include_canonical:
        names = [name for name in names if name != "canonical"]
    unknown = sorted(set(names) - set(WRAPPER_TEMPLATES))
    if unknown:
        raise SchemaError(f"Unknown wrapper templates: {unknown}")

    records: list[dict[str, object]] = []
    for row in questions.itertuples(index=False):
        for name in names:
            status = "canonical" if name == "canonical" else "validated"
            records.append(
                {
                    "variant_id": _variant_id(row.question_id, f"wrapper:{name}", row.question),
                    "question_id": row.question_id,
                    "subject": row.subject,
                    "question_text": row.question,
                    "choices": row.choices,
                    "answer_index": row.answer_index,
                    "variant": name,
                    "semantic_status": status,
                    "reviewer_notes": (
                        "" if name == "canonical" else "Deterministic wrapper; stem unchanged."
                    ),
                    "source": "deterministic-template",
                }
            )
    return records


def load_curated_paraphrases(
    path: str | Path, questions: pd.DataFrame
) -> pd.DataFrame:
    """Load curated text, requiring explicit reviewer evidence for validation.

    Required CSV columns follow `data/README.md`: `question_id`, `variant`,
    `paraphrased_question`, and `semantic_status`. Validation is never inferred:
    only an explicit `validated` value is treated as human-approved.
    """

    csv_path = Path(path)
    try:
        curated = pd.read_csv(csv_path)
    except OSError as exc:
        raise SchemaError(f"Could not read curated paraphrases {csv_path}: {exc}") from exc
    required = {"question_id", "variant", "paraphrased_question", "semantic_status"}
    missing = sorted(required - set(curated.columns))
    if missing:
        raise SchemaError(f"Curated paraphrase CSV is missing columns: {missing}")

    question_lookup = questions.set_index("question_id", drop=False)
    records: list[dict[str, object]] = []
    for row_number, row in curated.iterrows():
        question_id = str(row["question_id"]).strip()
        if question_id not in question_lookup.index:
            raise SchemaError(
                f"Curated row {row_number + 2} references unknown question_id {question_id!r}"
            )
        variant = str(row["variant"]).strip()
        if not variant or variant.lower() == "nan" or variant == "canonical":
            raise SchemaError(
                f"Curated row {row_number + 2} needs a non-canonical variant name"
            )
        text = str(row["paraphrased_question"]).strip()
        if not text or text.lower() == "nan":
            raise SchemaError(f"Curated row {row_number + 2} has no paraphrase text")
        requested_status = str(row["semantic_status"]).strip().lower()
        status = "validated" if requested_status == "validated" else "unreviewed"
        notes = str(row.get("reviewer_notes", "")).strip()
        if notes.lower() == "nan":
            notes = ""
        original = question_lookup.loc[question_id]
        records.append(
            {
                "variant_id": _variant_id(question_id, variant, text),
                "question_id": question_id,
                "subject": original["subject"],
                "question_text": text,
                "choices": original["choices"],
                "answer_index": original["answer_index"],
                "variant": variant,
                "semantic_status": status,
                "reviewer_notes": notes,
                "source": f"curated:{csv_path.name}",
            }
        )
    if not records:
        raise SchemaError("Curated paraphrase CSV contains no rows")
    return validate_variants(pd.DataFrame.from_records(records, columns=VARIANT_COLUMNS))


def build_variants(
    questions: pd.DataFrame, config: ParaphraseConfig
) -> pd.DataFrame:
    """Build canonical, wrapper, and/or curated variants for each question."""

    normalized = validate_questions(questions)
    frames: list[pd.DataFrame] = []

    if config.mode in {"templates", "templates+curated"} or (
        config.mode == "none" and config.include_canonical
    ):
        records = _template_variants(
            normalized,
            config.variants if config.mode != "none" else ("canonical",),
            config.include_canonical,
        )
        if records:
            frames.append(pd.DataFrame.from_records(records, columns=VARIANT_COLUMNS))
    elif config.mode == "curated" and config.include_canonical:
        records = _template_variants(normalized, ("canonical",), True)
        frames.append(pd.DataFrame.from_records(records, columns=VARIANT_COLUMNS))

    if config.mode in {"curated", "templates+curated"}:
        assert config.curated_path is not None
        frames.append(load_curated_paraphrases(config.curated_path, normalized))

    if not frames:
        raise SchemaError("Paraphrase configuration produced no variants")
    result = pd.concat(frames, ignore_index=True)
    result = result.drop_duplicates(subset="variant_id", keep="first")
    result = validate_variants(result)
    configured = set(config.variants)
    result = result.loc[result["variant"].isin(configured)].reset_index(drop=True)
    if result.duplicated(["question_id", "variant"]).any():
        duplicates = result.loc[
            result.duplicated(["question_id", "variant"], keep=False),
            ["question_id", "variant"],
        ].head(5)
        raise SchemaError(
            "Each question may have only one row per configured variant; duplicates include "
            + ", ".join(
                f"{row.question_id}/{row.variant}" for row in duplicates.itertuples(index=False)
            )
        )
    if config.require_validated:
        unapproved = result.loc[
            ~result["semantic_status"].isin({"canonical", "validated"}),
            ["question_id", "variant"],
        ]
        if not unapproved.empty:
            raise SchemaError(
                "Unvalidated curated paraphrases are not allowed when "
                "paraphrases.require_validated=true"
            )
        expected = {
            (question_id, variant)
            for question_id in normalized["question_id"]
            for variant in configured
        }
        actual = set(zip(result["question_id"], result["variant"], strict=True))
        missing_pairs = sorted(expected - actual)
        if missing_pairs:
            preview = ", ".join(
                f"{question_id}/{variant}" for question_id, variant in missing_pairs[:5]
            )
            raise SchemaError(
                f"Validated paraphrases are incomplete ({len(missing_pairs)} missing); "
                f"examples: {preview}"
            )
    if result.empty:
        raise SchemaError("No configured paraphrase variants were available")
    return result
