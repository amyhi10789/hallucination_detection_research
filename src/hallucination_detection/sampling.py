"""Deterministic, grouped sampling for the PopQA paraphrase pilot."""

from __future__ import annotations

import ast
import hashlib
import json
import random
from collections import Counter
from typing import Any, Iterable

from .paraphrases import SUPPORTED_RELATIONS, make_question_variants


def parse_answer_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
        except (ValueError, SyntaxError, json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, (list, tuple)):
            return [str(item) for item in parsed if str(item).strip()]
    return [text]


def _integer(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected numeric popularity, got {value!r}") from exc


def assign_popularity_bands(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Assign rank-based low/middle/high bands to avoid tied quantile issues."""

    prepared = [dict(row) for row in rows]
    prepared.sort(key=lambda row: (_integer(row["s_pop"]), str(row.get("id", ""))))
    count = len(prepared)
    if count < 3:
        raise ValueError("At least three eligible rows are required for popularity bands")

    for rank, row in enumerate(prepared):
        fraction = rank / count
        row["popularity_band"] = "low" if fraction < 1 / 3 else "middle" if fraction < 2 / 3 else "high"
    return prepared


def _band_targets(base_fact_count: int) -> dict[str, int]:
    if base_fact_count < 3:
        raise ValueError("Use at least three base facts to cover all popularity bands")
    base = base_fact_count // 3
    remainder = base_fact_count % 3
    targets = {"low": base, "middle": base, "high": base}
    if remainder >= 1:
        targets["low"] += 1
    if remainder >= 2:
        targets["high"] += 1
    return targets


def _select_diverse(
    candidates: list[dict[str, Any]], count: int, rng: random.Random, used_relations: set[str]
) -> list[dict[str, Any]]:
    shuffled = candidates[:]
    rng.shuffle(shuffled)
    selected: list[dict[str, Any]] = []

    for prefer_new_relation in (True, False):
        for row in shuffled:
            relation = str(row["prop"]).strip().lower()
            if row in selected:
                continue
            if prefer_new_relation and relation in used_relations:
                continue
            selected.append(row)
            used_relations.add(relation)
            if len(selected) == count:
                return selected
    raise ValueError(f"Could select only {len(selected)} of {count} requested rows")


def _subject_is_eligible(subject: str, relation: str, exclude_ambiguous_single_word_subjects: bool) -> bool:
    """Apply a preregisterable ambiguity screen before sampling.

    A one-word title or person name can refer to many entities (for example,
    "Time" or "Patience"). Geographic capital/country relations are retained
    because single-token place names are expected and the requested relation
    itself usually disambiguates them.
    """

    if not exclude_ambiguous_single_word_subjects:
        return True
    if relation in {"capital", "capital of", "country"}:
        return True
    return len(subject.split()) >= 2


def build_paired_pilot(
    rows: Iterable[dict[str, Any]],
    *,
    base_fact_count: int,
    seed: int,
    excluded_relations: Iterable[str] = (),
    exclude_ambiguous_single_word_subjects: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select base facts and expand each into four paired wordings."""

    excluded = {value.strip().lower() for value in excluded_relations}
    eligible: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        relation = str(row.get("prop", "")).strip().lower()
        if relation not in SUPPORTED_RELATIONS or relation in excluded:
            continue
        subject = str(row.get("subj", "")).strip()
        if not str(row.get("question", "")).strip() or not subject:
            continue
        if not _subject_is_eligible(subject, relation, exclude_ambiguous_single_word_subjects):
            continue
        answers = parse_answer_list(row.get("possible_answers"))
        if not answers:
            answers = parse_answer_list(row.get("obj"))
        canonical = str(row.get("obj", "")).strip()
        if canonical and canonical not in answers:
            answers.insert(0, canonical)
        if not answers:
            continue
        row["_answers"] = answers
        eligible.append(row)

    banded = assign_popularity_bands(eligible)
    targets = _band_targets(base_fact_count)
    rng = random.Random(seed)
    used_relations: set[str] = set()
    selected: list[dict[str, Any]] = []
    for band in ("low", "middle", "high"):
        candidates = [row for row in banded if row["popularity_band"] == band]
        selected.extend(_select_diverse(candidates, targets[band], rng, used_relations))

    output_rows: list[dict[str, Any]] = []
    for row in selected:
        dataset_id = str(row.get("id", ""))
        fact_id = f"popqa-{dataset_id}"
        relation = str(row["prop"]).strip().lower()
        subject = str(row["subj"]).strip()
        display_subject = str(row.get("s_wiki_title") or subject).strip()
        original_question = str(row["question"]).strip()
        if display_subject != subject:
            original_question = original_question.replace(subject, display_subject, 1)
        variants = make_question_variants(
            original_question=original_question,
            subject=display_subject,
            relation=relation,
        )
        for variant in variants:
            prompt_id = f"{fact_id}--{variant.variant_id}"
            output_rows.append(
                {
                    "prompt_id": prompt_id,
                    "fact_id": fact_id,
                    "dataset": "akariasai/PopQA",
                    "dataset_id": dataset_id,
                    "subject": subject,
                    "display_subject": display_subject,
                    "subject_wikipedia_title": str(row.get("s_wiki_title", "")),
                    "relation": relation,
                    "canonical_answer": str(row.get("obj", row["_answers"][0])),
                    "acceptable_answers": row["_answers"],
                    "subject_popularity": _integer(row["s_pop"]),
                    "popularity_band": row["popularity_band"],
                    "variant_id": variant.variant_id,
                    "paraphrase_type": variant.paraphrase_type,
                    "question": variant.question,
                    "meaning_review_status": "pending",
                    "meaning_review_notes": "",
                    "question_sha256": hashlib.sha256(variant.question.encode("utf-8")).hexdigest(),
                }
            )

    summary = {
        "seed": seed,
        "base_fact_count": len(selected),
        "prompt_count": len(output_rows),
        "variants_per_fact": 4,
        "exclude_ambiguous_single_word_subjects": exclude_ambiguous_single_word_subjects,
        "band_counts_by_fact": dict(Counter(row["popularity_band"] for row in selected)),
        "relation_counts_by_fact": dict(Counter(str(row["prop"]).strip().lower() for row in selected)),
        "fact_ids": [f"popqa-{row.get('id', '')}" for row in selected],
    }
    return output_rows, summary
