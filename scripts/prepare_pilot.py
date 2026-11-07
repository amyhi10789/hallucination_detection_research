"""Create a deterministic 5-fact/20-prompt paired-paraphrase pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset

from hallucination_detection.io_utils import read_json, write_jsonl
from hallucination_detection.sampling import build_paired_pilot


ROOT = Path(__file__).resolve().parents[1]
HF_DATASET_CACHE = ROOT / ".cache" / "huggingface" / "datasets"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "pilot.json")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data" / "processed" / "pilot_questions.jsonl"
    )
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "data" / "processed" / "pilot_manifest.json"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    dataset = load_dataset(
        config["dataset"],
        split=config["dataset_split"],
        cache_dir=str(HF_DATASET_CACHE),
    )
    pilot_rows, summary = build_paired_pilot(
        dataset,
        base_fact_count=int(config["base_fact_count"]),
        seed=int(config["seed"]),
        excluded_relations=config.get("excluded_relations", []),
        exclude_ambiguous_single_word_subjects=bool(
            config.get("exclude_ambiguous_single_word_subjects", True)
        ),
    )

    expected_prompts = int(config["base_fact_count"]) * int(config["variants_per_fact"])
    if len(pilot_rows) != expected_prompts:
        raise RuntimeError(f"Expected {expected_prompts} prompts, created {len(pilot_rows)}")

    write_jsonl(args.output, pilot_rows)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps({"config": config, "summary": summary}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"Saved {summary['prompt_count']} prompts from {summary['base_fact_count']} base facts")
    print(f"Popularity bands: {summary['band_counts_by_fact']}")
    print(f"Relations: {summary['relation_counts_by_fact']}")
    print(f"Questions: {args.output}")
    print(f"Manifest: {args.manifest}")
    print("NEXT: two teammates must review meaning_review_status for all 20 prompts.")


if __name__ == "__main__":
    main()
