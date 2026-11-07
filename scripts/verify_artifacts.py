"""Verify saved pilot metadata and hidden-state tensors without rerunning the model."""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from pathlib import Path

from hallucination_detection.io_utils import read_jsonl


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses", type=Path, default=ROOT / "artifacts" / "pilot" / "responses.jsonl")
    args = parser.parse_args()

    from safetensors import safe_open

    rows = read_jsonl(args.responses)
    if not rows:
        raise ValueError("No response records found")

    failures: list[str] = []
    seen_prompt_ids: set[str] = set()
    group_counts: Counter[str] = Counter()
    for row in rows:
        prompt_id = str(row["prompt_id"])
        if prompt_id in seen_prompt_ids:
            failures.append(f"{prompt_id}: duplicate prompt ID")
        seen_prompt_ids.add(prompt_id)
        group_counts[str(row["fact_id"])] += 1

        expected_hash = hashlib.sha256(str(row["rendered_prompt"]).encode("utf-8")).hexdigest()
        if expected_hash != row.get("prompt_sha256"):
            failures.append(f"{prompt_id}: prompt hash mismatch")
        if row.get("extraction_input_includes_reference") is not False:
            failures.append(f"{prompt_id}: reference-input leakage flag is not false")
        if not row.get("first_token_consistent"):
            failures.append(f"{prompt_id}: first-token consistency failed")
        if len(row.get("input_ids", [])) != len(row.get("attention_mask", [])):
            failures.append(f"{prompt_id}: input/attention-mask length mismatch")
        if int(row["embedding_plus_layer_count"]) != int(row["configured_layer_count"]) + 1:
            failures.append(f"{prompt_id}: layer count is not configured layers + 1")

        tensor_path = ROOT / str(row["hidden_state_path"])
        if not tensor_path.exists():
            failures.append(f"{prompt_id}: missing {tensor_path}")
            continue
        with safe_open(str(tensor_path), framework="pt", device="cpu") as handle:
            if "hidden_states" not in handle.keys():
                failures.append(f"{prompt_id}: hidden_states tensor is missing")
                continue
            tensor = handle.get_tensor("hidden_states")
            if list(tensor.shape) != row.get("hidden_state_shape"):
                failures.append(f"{prompt_id}: tensor shape does not match metadata")
            if str(tensor.dtype).replace("torch.", "") != row.get("hidden_state_dtype"):
                failures.append(f"{prompt_id}: tensor dtype does not match metadata")

    unexpected_group_sizes = {fact_id: count for fact_id, count in group_counts.items() if count != 4}
    if unexpected_group_sizes:
        failures.append(f"Expected four variants per fact: {unexpected_group_sizes}")

    if failures:
        print("Verification FAILED:")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)
    print(f"Verification passed for {len(rows)} prompts in {len(group_counts)} paraphrase groups")


if __name__ == "__main__":
    main()
