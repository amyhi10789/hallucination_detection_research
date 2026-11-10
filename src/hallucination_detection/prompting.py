"""Deterministic four-choice prompt rendering."""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from .paraphrases import WRAPPER_TEMPLATES
from .schemas import SchemaError, coerce_choices, validate_variants


ANSWER_LETTERS = ("A", "B", "C", "D")
ANSWER_ONLY_INSTRUCTION = "Respond with only the letter A, B, C, or D."


def render_prompt(
    question_text: str,
    choices: Any,
    template_name: str = "canonical",
) -> str:
    """Render a question without modifying or reordering its four choices."""

    if template_name not in WRAPPER_TEMPLATES:
        raise SchemaError(f"Unknown prompt wrapper: {template_name!r}")
    question = str(question_text).strip()
    if not question:
        raise SchemaError("question_text cannot be blank")
    normalized_choices = coerce_choices(choices)
    wrapped_question = WRAPPER_TEMPLATES[template_name].format(question=question)
    option_lines = "\n".join(
        f"{letter}. {choice}"
        for letter, choice in zip(ANSWER_LETTERS, normalized_choices, strict=True)
    )
    return f"{wrapped_question}\n\n{option_lines}\n\n{ANSWER_ONLY_INSTRUCTION}"


def render_prompt_record(record: Mapping[str, Any]) -> str:
    """Render one mapping that follows the variant schema."""

    return render_prompt(
        record["question_text"],
        record["choices"],
        record["variant"] if record["variant"] in WRAPPER_TEMPLATES else "canonical",
    )


def add_prompts(variants: pd.DataFrame) -> pd.DataFrame:
    """Return validated variants with an appended deterministic `prompt` column."""

    normalized = validate_variants(variants)
    result = normalized.copy()
    result["prompt"] = [
        render_prompt(
            row.question_text,
            row.choices,
            row.variant if row.variant in WRAPPER_TEMPLATES else "canonical",
        )
        for row in result.itertuples(index=False)
    ]
    return result
