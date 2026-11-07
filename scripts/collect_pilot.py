"""Generate answers and save final-prompt-token hidden states for the pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hallucination_detection.io_utils import read_json, read_jsonl, write_jsonl
from hallucination_detection.labeling import label_answer


ROOT = Path(__file__).resolve().parents[1]
HF_MODEL_CACHE = ROOT / ".cache" / "huggingface" / "hub"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "pilot.json")
    parser.add_argument(
        "--questions", type=Path, default=ROOT / "data" / "processed" / "pilot_questions.jsonl"
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "pilot")
    parser.add_argument(
        "--fact-limit",
        type=int,
        default=None,
        help="Run only the first N complete fact groups for a smoke test.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allow-unreviewed",
        action="store_true",
        help="Permit pilot collection before meaning-equivalence review (debugging only).",
    )
    return parser.parse_args()


def _last_non_padding_index(attention_mask: Any) -> int:
    positions = attention_mask[0].nonzero(as_tuple=False).flatten()
    if positions.numel() == 0:
        raise ValueError("The prompt attention mask contains no non-padding token")
    return int(positions[-1].item())


def _render_prompt(tokenizer: Any, instruction: str, question: str) -> str:
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def main() -> None:
    args = parse_args()
    args.output_dir = (
        args.output_dir.resolve()
        if args.output_dir.is_absolute()
        else (ROOT / args.output_dir).resolve()
    )
    config = read_json(args.config)
    questions = read_jsonl(args.questions)
    if not questions:
        raise ValueError(f"No questions found in {args.questions}")
    if args.fact_limit is not None:
        if args.fact_limit < 1:
            raise ValueError("--fact-limit must be at least 1")
        selected_fact_ids = list(dict.fromkeys(str(row["fact_id"]) for row in questions))[
            : args.fact_limit
        ]
        questions = [row for row in questions if str(row["fact_id"]) in selected_fact_ids]
    unreviewed = [row["prompt_id"] for row in questions if row.get("meaning_review_status") != "approved"]
    if unreviewed and not args.allow_unreviewed:
        raise ValueError(
            f"{len(unreviewed)} prompts have not been approved for meaning equivalence. "
            "Review the JSONL first, or use --allow-unreviewed for a plumbing-only run."
        )

    metadata_path = args.output_dir / "responses.jsonl"
    hidden_dir = args.output_dir / "hidden_states"
    if metadata_path.exists() and not args.overwrite:
        raise FileExistsError(f"{metadata_path} already exists; pass --overwrite to replace the pilot run")

    # Direct HTTP is more reliable than the optional Xet transport on some
    # school/Windows networks. Existing partial downloads are still resumed.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

    import torch
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    set_seed(int(config["seed"]))
    model_name = str(config["model"])
    revision = str(config.get("model_revision", "main"))
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, revision=revision, cache_dir=str(HF_MODEL_CACHE)
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        cache_dir=str(HF_MODEL_CACHE),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    hidden_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    for number, row in enumerate(questions, start=1):
        rendered_prompt = _render_prompt(
            tokenizer, str(config["prompt_instruction"]), str(row["question"])
        )
        inputs = tokenizer(rendered_prompt, return_tensors="pt", add_special_tokens=False)
        inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
        final_prompt_index = _last_non_padding_index(inputs["attention_mask"])

        with torch.inference_mode():
            forward = model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
            if forward.hidden_states is None:
                raise RuntimeError("Model did not return hidden states")
            hidden_states = torch.stack(
                [state[0, final_prompt_index, :].detach().to(torch.float16).cpu() for state in forward.hidden_states]
            )
            forward_first_token = int(forward.logits[0, final_prompt_index, :].argmax().item())
            generated = model.generate(
                **inputs,
                do_sample=bool(config["do_sample"]),
                max_new_tokens=int(config["max_new_tokens"]),
                temperature=float(config["temperature"]),
                top_p=float(config["top_p"]),
                top_k=int(config["top_k"]),
                repetition_penalty=float(config["repetition_penalty"]),
                pad_token_id=tokenizer.eos_token_id,
            )

        prompt_length = int(inputs["input_ids"].shape[1])
        generated_token_ids = generated[0, prompt_length:].detach().cpu()
        answer = tokenizer.decode(generated_token_ids, skip_special_tokens=True).strip()
        generated_first_token = int(generated_token_ids[0].item()) if generated_token_ids.numel() else None
        first_token_consistent = generated_first_token == forward_first_token

        hidden_path = hidden_dir / f"{row['prompt_id']}.safetensors"
        save_file({"hidden_states": hidden_states.contiguous()}, str(hidden_path))
        automatic = label_answer(answer, row["acceptable_answers"])

        record = {
            **row,
            "rendered_prompt": rendered_prompt,
            "prompt_sha256": hashlib.sha256(rendered_prompt.encode("utf-8")).hexdigest(),
            "input_ids": inputs["input_ids"][0].detach().cpu().tolist(),
            "attention_mask": inputs["attention_mask"][0].detach().cpu().tolist(),
            "final_prompt_token_index": final_prompt_index,
            "final_prompt_token_id": int(inputs["input_ids"][0, final_prompt_index].item()),
            "final_prompt_token_text": tokenizer.decode(
                [int(inputs["input_ids"][0, final_prompt_index].item())]
            ),
            "generated_token_ids": generated_token_ids.tolist(),
            "generated_answer": answer,
            "automatic_label": automatic.label,
            "matched_alias": automatic.matched_alias,
            "needs_manual_review": automatic.needs_manual_review,
            "reviewed_label": None,
            "reviewer_1": "",
            "reviewer_1_label": "",
            "reviewer_2": "",
            "reviewer_2_label": "",
            "adjudication_notes": "",
            "hidden_state_path": hidden_path.relative_to(ROOT).as_posix(),
            "hidden_state_shape": list(hidden_states.shape),
            "hidden_state_dtype": str(hidden_states.dtype).replace("torch.", ""),
            "embedding_plus_layer_count": len(forward.hidden_states),
            "configured_layer_count": int(model.config.num_hidden_layers),
            "forward_first_token_id": forward_first_token,
            "generated_first_token_id": generated_first_token,
            "first_token_consistent": first_token_consistent,
            "extraction_input_includes_reference": False,
            "model_name": model_name,
            "requested_model_revision": revision,
            "resolved_model_revision": getattr(model.config, "_commit_hash", None) or revision,
            "tokenizer_name": tokenizer.name_or_path,
            "device": str(device),
            "torch_version": torch.__version__,
            "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        records.append(record)
        print(
            f"[{number:02d}/{len(questions):02d}] {row['prompt_id']}: "
            f"{automatic.label}; hidden={list(hidden_states.shape)}; "
            f"first-token-check={first_token_consistent}"
        )

    write_jsonl(metadata_path, records)
    failures = sum(not record["first_token_consistent"] for record in records)
    print(f"Saved {len(records)} response records to {metadata_path}")
    print(f"First-token consistency failures: {failures}")
    if failures:
        raise RuntimeError("At least one first-token consistency check failed")


if __name__ == "__main__":
    main()
