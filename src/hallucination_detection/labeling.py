"""Conservative automatic labels for short factual answers."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable


_ABSTENTION_PATTERNS = (
    r"\bi do not know\b",
    r"\bi don't know\b",
    r"\bnot sure\b",
    r"\bunknown\b",
    r"\bcannot answer\b",
    r"\bcan't answer\b",
)


@dataclass(frozen=True)
class LabelResult:
    label: str
    matched_alias: str | None
    normalized_generation: str
    needs_manual_review: bool


def normalize_answer(text: str) -> str:
    """Normalize case, Unicode, punctuation, articles, and whitespace."""

    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def _contains_whole_phrase(text: str, phrase: str) -> bool:
    if not phrase:
        return False
    return re.search(rf"(?:^|\s){re.escape(phrase)}(?:$|\s)", text) is not None


def label_answer(generation: str, acceptable_answers: Iterable[str]) -> LabelResult:
    """Create an automatic pilot label while flagging mismatches for review."""

    raw_lower = unicodedata.normalize("NFKC", generation).lower()
    normalized_generation = normalize_answer(generation)

    if any(re.search(pattern, raw_lower) for pattern in _ABSTENTION_PATTERNS):
        return LabelResult("abstain", None, normalized_generation, False)

    aliases = [answer for answer in acceptable_answers if str(answer).strip()]
    aliases.sort(key=lambda value: len(normalize_answer(str(value))), reverse=True)
    for alias in aliases:
        normalized_alias = normalize_answer(str(alias))
        if normalized_generation == normalized_alias:
            return LabelResult("correct", str(alias), normalized_generation, False)
        if _contains_whole_phrase(normalized_generation, normalized_alias):
            alias_token_count = len(normalized_alias.split())
            generation_token_count = len(normalized_generation.split())
            # A broad single-token alias such as "Church" must not mark a long,
            # contradictory answer as correct. Short scaffolding ("It is actor")
            # remains acceptable; longer single-token cases go to human review.
            safe_containment = alias_token_count >= 2 or generation_token_count <= alias_token_count + 3
            if safe_containment:
                return LabelResult("correct", str(alias), normalized_generation, False)

    return LabelResult("incorrect", None, normalized_generation, True)
