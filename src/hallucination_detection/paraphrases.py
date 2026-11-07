"""Controlled, relation-specific paraphrases for PopQA.

The variants intentionally change surface wording while preserving the subject,
requested relation, and expected answer. Every generated group still requires a
brief human meaning-equivalence review before it is used in an experiment.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class QuestionVariant:
    variant_id: str
    paraphrase_type: str
    question: str


_TEMPLATES: dict[str, tuple[str, str, str]] = {
    "occupation": (
        "What does {subject} do for a living?",
        "Which occupation is {subject} known for?",
        "What profession does {subject} have?",
    ),
    "place of birth": (
        "What city is {subject}'s birthplace?",
        "Which city was {subject} born in?",
        "Name the city where {subject} was born.",
    ),
    "genre": (
        "Which genre does {subject} belong to?",
        "How is {subject} classified by genre?",
        "What is the genre of {subject}?",
    ),
    "father": (
        "Who is {subject}'s father?",
        "What is the name of the father of {subject}?",
        "Name the person who is the father of {subject}.",
    ),
    "country": (
        "Which country is {subject} in?",
        "{subject} is located in which country?",
        "Name the country containing {subject}.",
    ),
    "producer": (
        "Who produced {subject}?",
        "Who is credited as the producer of {subject}?",
        "Name the producer of {subject}.",
    ),
    "director": (
        "Who directed {subject}?",
        "Who is credited as the director of {subject}?",
        "Name the director of {subject}.",
    ),
    "capital of": (
        "{subject} is the capital of which place?",
        "What place has {subject} as its capital?",
        "Name the place whose capital is {subject}.",
    ),
    "screenwriter": (
        "Who wrote the screenplay for {subject}?",
        "Who is credited as the screenwriter of {subject}?",
        "Name the screenwriter for {subject}.",
    ),
    "composer": (
        "Who composed {subject}?",
        "Who is credited as the composer of {subject}?",
        "Name the composer of {subject}.",
    ),
    "religion": (
        "Which religion is {subject} associated with?",
        "What religious faith does {subject} follow?",
        "What is the religious affiliation of {subject}?",
    ),
    "sport": (
        "Which sport is {subject} known for playing?",
        "What athletic sport does {subject} participate in?",
        "{subject} competes in which sport?",
    ),
    "author": (
        "Who wrote {subject}?",
        "Who is credited as the author of {subject}?",
        "Name the author of {subject}.",
    ),
    "mother": (
        "Who is {subject}'s mother?",
        "What is the name of the mother of {subject}?",
        "Name the person who is the mother of {subject}.",
    ),
    "capital": (
        "Which city is the capital of {subject}?",
        "What city serves as {subject}'s capital?",
        "Name the capital city of {subject}.",
    ),
}

SUPPORTED_RELATIONS = frozenset(_TEMPLATES)


def _comparison_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def make_question_variants(
    *, original_question: str, subject: str, relation: str
) -> list[QuestionVariant]:
    """Return one original question and three controlled paraphrases.

    Raises:
        ValueError: if the relation is unsupported or four distinct wordings
            cannot be constructed.
    """

    normalized_relation = relation.strip().lower()
    if normalized_relation not in _TEMPLATES:
        raise ValueError(
            f"Unsupported PopQA relation {relation!r}. "
            f"Supported relations: {sorted(SUPPORTED_RELATIONS)}"
        )

    candidates = [
        QuestionVariant("original", "original", original_question.strip()),
        QuestionVariant(
            "lexical", "lexical_substitution", _TEMPLATES[normalized_relation][0].format(subject=subject)
        ),
        QuestionVariant(
            "syntactic", "syntactic_reordering", _TEMPLATES[normalized_relation][1].format(subject=subject)
        ),
        QuestionVariant(
            "concise", "concise_instruction", _TEMPLATES[normalized_relation][2].format(subject=subject)
        ),
    ]

    seen: set[str] = set()
    for variant in candidates:
        key = _comparison_key(variant.question)
        if not key or key in seen:
            raise ValueError(
                f"Relation {relation!r} produced a duplicate/empty wording: {variant.question!r}"
            )
        if _comparison_key(subject) not in key:
            raise ValueError(
                f"Paraphrase dropped the subject {subject!r}: {variant.question!r}"
            )
        seen.add(key)

    return candidates
