"""Export model answers to a CSV for two independent human reviewers."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from hallucination_detection.io_utils import read_jsonl


ROOT = Path(__file__).resolve().parents[1]
FIELDS = [
    "prompt_id",
    "fact_id",
    "variant_id",
    "question",
    "canonical_answer",
    "acceptable_answers",
    "generated_answer",
    "automatic_label",
    "needs_manual_review",
    "reviewer_1",
    "reviewer_1_label",
    "reviewer_2",
    "reviewer_2_label",
    "reviewed_label",
    "adjudication_notes",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses", type=Path, default=ROOT / "artifacts" / "pilot" / "responses.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "pilot_review.csv")
    args = parser.parse_args()

    rows = read_jsonl(args.responses)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            prepared = dict(row)
            prepared["acceptable_answers"] = json.dumps(
                row.get("acceptable_answers", []), ensure_ascii=False
            )
            writer.writerow(prepared)
    print(f"Saved {len(rows)} review rows to {args.output}")


if __name__ == "__main__":
    main()
