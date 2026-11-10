from pathlib import Path
from dataclasses import replace

import pytest

from hallucination_detection.config import load_config
from hallucination_detection.data import load_synthetic_questions
from hallucination_detection.paraphrases import build_variants
from hallucination_detection.prompting import add_prompts
from hallucination_detection.schemas import SchemaError


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_all_repository_profiles_are_valid():
    for name in ("smoke", "pilot", "full", "synthetic"):
        config = load_config(REPO_ROOT / "configs" / f"{name}.toml")
        assert config.run.name == name


def test_template_variants_preserve_choices_and_gold_answer():
    config = load_config(REPO_ROOT / "configs" / "synthetic.toml")
    questions = load_synthetic_questions(size=8, seed=7, subjects=("a", "b"))
    variants = add_prompts(build_variants(questions, config.paraphrases))

    assert set(variants["variant"]) == {"canonical", "direct", "exam"}
    assert variants.groupby("question_id")["answer_index"].nunique().max() == 1
    assert variants.groupby("question_id")["choices"].nunique().max() == 1
    assert variants["prompt"].str.contains("Respond with only the letter").all()


def test_confirmatory_profile_rejects_incomplete_curated_paraphrases(tmp_path):
    config = load_config(REPO_ROOT / "configs" / "synthetic.toml")
    questions = load_synthetic_questions(size=2, seed=7, subjects=("a",))
    curated = tmp_path / "curated.csv"
    curated.write_text(
        "question_id,variant,paraphrased_question,semantic_status,reviewer_notes\n"
        f"{questions.iloc[0].question_id},paraphrase_1,A reviewed rewrite?,validated,checked\n",
        encoding="utf-8",
    )
    strict = replace(
        config.paraphrases,
        mode="curated",
        variants=("canonical", "paraphrase_1"),
        curated_path=curated,
        require_validated=True,
    )

    with pytest.raises(SchemaError, match="incomplete"):
        build_variants(questions, strict)

