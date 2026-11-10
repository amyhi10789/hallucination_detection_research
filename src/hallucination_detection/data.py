"""Question loading and deterministic stratified sampling."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .config import DataConfig, ExperimentConfig
from .schemas import QUESTION_COLUMNS, SchemaError, coerce_answer_index, coerce_choices

if TYPE_CHECKING:
    from datasets import Dataset


def _stable_question_id(
    source: str,
    subject: str,
    question: str,
    choices: tuple[str, str, str, str],
    answer_index: int,
) -> str:
    payload = json.dumps(
        {
            "source": source,
            "subject": subject,
            "question": question,
            "choices": choices,
            "answer_index": answer_index,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return "q_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def _largest_remainder_quotas(counts: pd.Series, target: int) -> dict[str, int]:
    raw = counts.astype(float) * (target / int(counts.sum()))
    quotas = np.floor(raw).astype(int)
    remaining = target - int(quotas.sum())
    order = sorted(counts.index, key=lambda key: (-(raw[key] - quotas[key]), str(key)))
    for subject in order[:remaining]:
        quotas[subject] += 1
    return {str(subject): int(quotas[subject]) for subject in counts.index}


def deterministic_stratified_sample(
    questions: pd.DataFrame, max_questions: int | None, seed: int
) -> pd.DataFrame:
    """Sample proportionally by subject, independent of incoming row order."""

    if max_questions is None or max_questions >= len(questions):
        return questions.sort_values(["subject", "question_id"]).reset_index(drop=True)
    if max_questions <= 0:
        raise ValueError("max_questions must be positive")

    counts = questions.groupby("subject", sort=True).size()
    quotas = _largest_remainder_quotas(counts, max_questions)
    selected: list[pd.DataFrame] = []
    for subject in sorted(quotas):
        count = quotas[subject]
        if count == 0:
            continue
        group = questions.loc[questions["subject"] == subject].sort_values("question_id")
        subject_seed = int.from_bytes(
            hashlib.sha256(f"{seed}:{subject}".encode("utf-8")).digest()[:8], "big"
        )
        rng = np.random.default_rng(subject_seed)
        positions = np.sort(rng.choice(len(group), size=count, replace=False))
        selected.append(group.iloc[positions])
    return (
        pd.concat(selected, ignore_index=True)
        .sort_values(["subject", "question_id"])
        .reset_index(drop=True)
    )


def _normalize_dataset(dataset: Dataset, config: DataConfig) -> pd.DataFrame:
    required = {"question", "subject", "choices", "answer"}
    missing = sorted(required - set(dataset.column_names))
    if missing:
        raise SchemaError(
            f"{config.dataset_id}/{config.dataset_config} is missing fields: {missing}"
        )

    records: list[dict[str, object]] = []
    source = f"{config.dataset_id}:{config.dataset_config}"
    for row in dataset:
        question = str(row["question"]).strip()
        subject = str(row["subject"]).strip()
        choices = coerce_choices(row["choices"])
        answer_index = coerce_answer_index(row["answer"])
        if not question or not subject:
            raise SchemaError("Dataset contains a blank question or subject")
        records.append(
            {
                "question_id": _stable_question_id(
                    source, subject, question, choices, answer_index
                ),
                "source": source,
                "source_split": config.split,
                "subject": subject,
                "question": question,
                "choices": choices,
                "answer_index": answer_index,
            }
        )
    frame = pd.DataFrame.from_records(records, columns=QUESTION_COLUMNS)
    if frame["question_id"].duplicated().any():
        frame = frame.drop_duplicates(subset=list(QUESTION_COLUMNS[1:]), keep="first")
    if frame["question_id"].duplicated().any():
        raise SchemaError("Content-derived MMLU question IDs are not unique")
    return frame.reset_index(drop=True)


def load_mmlu_questions(config: DataConfig, seed: int = 42) -> pd.DataFrame:
    """Load `cais/mmlu` (normally config `all`) and normalize its four fields."""

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' package is required to load MMLU. Install project dependencies."
        ) from exc

    kwargs: dict[str, object] = {
        "path": config.dataset_id,
        "name": config.dataset_config,
        "split": config.split,
        "trust_remote_code": config.trust_remote_code,
    }
    if config.revision is not None:
        kwargs["revision"] = config.revision
    if config.cache_dir is not None:
        kwargs["cache_dir"] = str(config.cache_dir)
    dataset = load_dataset(**kwargs)
    frame = _normalize_dataset(dataset, config)
    if config.subjects:
        requested = set(config.subjects)
        available = set(frame["subject"])
        missing = sorted(requested - available)
        if missing:
            raise SchemaError(f"Requested subjects not found in dataset: {missing}")
        frame = frame.loc[frame["subject"].isin(requested)].copy()
    if frame.empty:
        raise SchemaError("No MMLU questions remain after filtering")
    return deterministic_stratified_sample(frame, config.max_questions, seed)


def load_synthetic_questions(
    size: int = 24,
    seed: int = 42,
    subjects: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Create a small deterministic four-choice dataset for offline tests."""

    if size <= 0:
        raise ValueError("size must be positive")
    subject_names = subjects or ("mathematics", "science", "history")
    records: list[dict[str, object]] = []
    for index in range(size):
        subject = subject_names[index % len(subject_names)]
        left = index + 2
        right = (index % 5) + 1
        correct = left + right
        choices = tuple(str(correct + offset) for offset in (-1, 0, 1, 2))
        answer_index = 1
        question = f"What is {left} + {right}?"
        records.append(
            {
                "question_id": _stable_question_id(
                    "synthetic", subject, question, choices, answer_index
                ),
                "source": "synthetic",
                "source_split": "test",
                "subject": subject,
                "question": question,
                "choices": choices,
                "answer_index": answer_index,
            }
        )
    frame = pd.DataFrame.from_records(records, columns=QUESTION_COLUMNS)
    return deterministic_stratified_sample(frame, size, seed)


def load_questions(
    config: DataConfig | ExperimentConfig, seed: int | None = None
) -> pd.DataFrame:
    """Load either configured MMLU data or the deterministic test fixture."""

    if isinstance(config, ExperimentConfig):
        data_config = config.data
        resolved_seed = config.run.seed if seed is None else seed
    else:
        data_config = config
        resolved_seed = 42 if seed is None else seed
    if data_config.backend == "synthetic":
        size = data_config.max_questions or data_config.synthetic_size
        frame = load_synthetic_questions(size, resolved_seed, data_config.subjects)
        return deterministic_stratified_sample(
            frame, data_config.max_questions, resolved_seed
        )
    return load_mmlu_questions(data_config, resolved_seed)
