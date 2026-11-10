from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from hallucination_detection.config import load_config
from hallucination_detection.data import load_questions


STYLE_INSTRUCTIONS = {
    "paraphrase_1": "Use different sentence structure and common vocabulary.",
    "paraphrase_2": "Rewrite concisely while preserving every condition and qualification.",
    "paraphrase_3": "Use an alternate natural phrasing without adding explanation.",
}


def draft_paraphrases(
    config_path: str | Path,
    output_path: str | Path,
    *,
    limit: int | None = None,
    max_new_tokens: int = 128,
) -> Path:
    """Draft unreviewed paraphrases with the configured local Hugging Face model."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config_file = Path(config_path).resolve()
    config = load_config(config_file)
    questions = load_questions(config)
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        questions = questions.head(limit).copy()

    destination = Path(output_path).resolve()
    if destination.exists() and destination.stat().st_size > 0:
        raise FileExistsError(
            f"Refusing to overwrite {destination}. Move it or choose another output path."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)

    device_name = str(config.model.device).lower()
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    dtype_name = str(config.model.dtype).lower()
    if dtype_name == "auto":
        dtype = torch.float16 if device.type == "cuda" else torch.float32
    else:
        dtype = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }[dtype_name]

    tokenizer = AutoTokenizer.from_pretrained(
        config.model.model_id,
        revision=config.model.revision or "main",
        trust_remote_code=False,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config.model.model_id,
        revision=config.model.revision or "main",
        torch_dtype=dtype,
        trust_remote_code=False,
    ).to(device)
    model.eval()

    tasks: list[dict[str, Any]] = []
    for row in questions.itertuples(index=False):
        for variant, style in STYLE_INSTRUCTIONS.items():
            tasks.append(
                {
                    "question_id": row.question_id,
                    "subject": row.subject,
                    "original_question": row.question,
                    "choices": row.choices,
                    "answer_index": row.answer_index,
                    "variant": variant,
                    "instruction": _draft_instruction(row.question, style),
                }
            )

    rows: list[dict[str, Any]] = []
    batch_size = int(config.model.batch_size)
    for start in range(0, len(tasks), batch_size):
        task_batch = tasks[start : start + batch_size]
        encoded_rows = []
        for task in task_batch:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You rewrite research benchmark questions. Never answer the question "
                        "and never change its meaning."
                    ),
                },
                {"role": "user", "content": task["instruction"]},
            ]
            encoded = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
            )
            input_ids = list(encoded["input_ids"])
            attention_mask = list(encoded.get("attention_mask", [1] * len(input_ids)))
            encoded_rows.append(
                {
                    "input_ids": input_ids[-config.model.max_input_tokens :],
                    "attention_mask": attention_mask[-config.model.max_input_tokens :],
                }
            )
        batch = tokenizer.pad(encoded_rows, padding=True, return_tensors="pt")
        batch = {name: tensor.to(device) for name, tensor in batch.items()}
        with torch.inference_mode():
            generated = model.generate(
                **batch,
                do_sample=False,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        outputs = tokenizer.batch_decode(
            generated[:, batch["input_ids"].shape[1] :], skip_special_tokens=True
        )
        for task, output in zip(task_batch, outputs, strict=True):
            candidate = clean_candidate(output)
            warnings = anchor_warnings(task["original_question"], candidate)
            rows.append(
                {
                    "question_id": task["question_id"],
                    "variant": task["variant"],
                    "paraphrased_question": candidate,
                    "semantic_status": "unreviewed",
                    "reviewer_notes": "; ".join(warnings),
                    "subject": task["subject"],
                    "original_question": task["original_question"],
                    "choices": json.dumps(task["choices"], ensure_ascii=False),
                    "answer_index": task["answer_index"],
                    "correct_answer": task["choices"][task["answer_index"]],
                }
            )

    pd.DataFrame.from_records(rows).to_csv(destination, index=False)
    return destination


def clean_candidate(text: str) -> str:
    candidate = str(text).strip()
    candidate = re.sub(
        r"^(?:rewritten|paraphrased)\s+question\s*:\s*",
        "",
        candidate,
        flags=re.IGNORECASE,
    ).strip()
    return candidate.splitlines()[0].strip().strip('"“”') if candidate else ""


def anchor_warnings(original: str, candidate: str) -> list[str]:
    warnings: list[str] = []
    if not candidate:
        return ["EMPTY OUTPUT — rewrite manually"]
    if candidate.casefold() == original.strip().casefold():
        warnings.append("UNCHANGED — rewrite manually")
    original_numbers = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", original))
    candidate_numbers = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", candidate))
    missing_numbers = sorted(original_numbers - candidate_numbers)
    if missing_numbers:
        warnings.append("CHECK MISSING NUMBER(S): " + ", ".join(missing_numbers))
    anchors = {"not", "except", "least", "false", "incorrect", "never"}
    original_words = set(re.findall(r"[a-z]+", original.casefold()))
    candidate_words = set(re.findall(r"[a-z]+", candidate.casefold()))
    missing_anchors = sorted((original_words & anchors) - candidate_words)
    if missing_anchors:
        warnings.append("CHECK MISSING LOGIC WORD(S): " + ", ".join(missing_anchors))
    return warnings


def _draft_instruction(question: str, style: str) -> str:
    return (
        "Rewrite the question below without changing its meaning, facts, numbers, entities, "
        "negation, scope, or correct answer. Do not solve it. Do not mention answer choices. "
        "Output exactly one rewritten question and nothing else. "
        f"{style}\n\nQuestion: {question}"
    )

